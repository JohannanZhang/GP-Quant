"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""


from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement

from custom_function import _Function
from gplearn.utils import check_random_state
import pandas as pd

class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 ts_function_set,
                 d_ls,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 # metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None):

        self.ts_function_set = ts_function_set
        self.d_ls = d_ls
        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        # self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set) + len(self.ts_function_set))
        if function < len(self.function_set):
            function = self.function_set[function]
        else:
            function = make_ts_function(self.ts_function_set[function - len(self.function_set)],
                                        self.d_ls, random_state)

        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set) + len(self.ts_function_set)
            choice = random_state.randint(choice)
            if (depth < max_depth) and (method == 'full' or
                                        choice < len(self.function_set) + len(self.ts_function_set)):
                if choice < len(self.function_set):
                    function_id = random_state.randint(len(self.function_set))
                    function = self.function_set[function_id]
                else:
                    # 随机选择时序算子的实例对象，并为属性d、name赋值
                    function_id = random_state.randint(len(self.ts_function_set))
                    function = make_ts_function(self.ts_function_set[function_id],
                                                self.d_ls, random_state)
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    # terminal = random_state.randint(self.n_features + 1)
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')

                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None


    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_days, n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        factor : array-like, shape = [n_days, n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        # 针对有常数项的情况
        # if isinstance(node, float):
        #     return np.repeat(node, X.shape[0])
        # 若公式树开头就为单变量，则将其导出
        if isinstance(node, int):
            return X[:, :, node]

        apply_stack = []

        for node in self.program:
            X = X.copy()
            if isinstance(node, _Function):
                apply_stack.append([node])

            else:
                apply_stack[-1].append(node)
            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = np.zeros((X.shape[0], X.shape[1]))
                found = False
                # 遍历列表并比较name属性
                for obj in self.function_set:
                    if obj.name == function.name:
                        found = True

                # 针对截面算子实例对象，单变量情况时：
                if found and function.arity == 1:
                    if isinstance(apply_stack[-1][1], int):
                        terminals = X[:, :, apply_stack[-1][1]]
                    else:
                        terminals = apply_stack[-1][1]
                    for i in range(terminals.shape[0]):
                        for j in range(terminals.shape[1]):
                            terminals[i, j] = function(terminals[i, j])

                # 针对截面算子实例对象，双变量情况时：
                if found and function.arity == 2:
                    terminals = np.zeros((X.shape[0], X.shape[1]))
                    if isinstance(apply_stack[-1][1], int):
                        terminalsA = X[:, :, apply_stack[-1][1]]
                    else:
                        terminalsA = apply_stack[-1][1]

                    if isinstance(apply_stack[-1][2], int):
                        terminalsB = X[:, :, apply_stack[-1][2]]
                    else:
                        terminalsB = apply_stack[-1][2]

                    for i in range(terminalsA.shape[0]):
                        for j in range(terminalsA.shape[1]):
                            terminals[i, j] = function(terminalsA[i, j], terminalsB[i, j],)

                # 针对时序算子实例对象，单变量情况时：
                if not found and function.arity == 1:
                    if isinstance(apply_stack[-1][1], int):
                        terminals = X[:, :, apply_stack[-1][1]]
                    else:
                        terminals = apply_stack[-1][1]
                    for i in range(terminals.shape[1]):
                        terminals[:, i] = function(terminals[:, i])

                # # 针对时序算子实例对象，双变量情况时：
                # if not found and function.arity == 2:
                #     terminals = np.zeros((X.shape[0], X.shape[1]))
                #     if isinstance(apply_stack[-1][1], int):
                #         terminalA = X[:, :, apply_stack[-1][1]]
                #     else:
                #         terminalA = apply_stack[-1][1]
                #     if isinstance(apply_stack[-1][2], int):
                #         terminalB = X[:, :, apply_stack[-1][1]]
                #     else:
                #         terminalB = apply_stack[-1][1]
                #
                #     for i in range(terminals.shape[1]):
                #         terminals[:, i] = function(terminalA[:, i], terminalB[:, i])

                intermediate_result = terminals


                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    factor_result = np.empty_like(intermediate_result)
                    for i in range(factor_result.shape[0]):
                        row = intermediate_result[i, :]
                        rank = np.argsort(row)  # 计算排名
                        rank_normalized_row = rank.argsort() / (intermediate_result.shape[1] - 1)  # 排名标准化
                        factor_result[i, :] = rank_normalized_row
                    return factor_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]



    # 比对在固定年份下，基于因子值的十分组多头组合，相对于当年中证500指数的超额信息率
    def raw_fitness(self, X, Y, year):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_days, n_samples]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Y : array-like, shape = [n_days, n_samples]
            Target values.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        factor = self.execute(X)
        correlation_coefficients = []
        for i in range(factor.shape[0]-2):
            col_X = factor[i, :]  # 获取X的第i列
            # t日结束计算因子值，t+1进行交易，t+2得到收益率，所以关注Y的第i+2行与factor第i行的相关性
            col_Y = Y[i + 2, :]
            # 计算相关系数
            correlation = np.corrcoef(col_X, col_Y)[0, 1]
            # 将相关系数添加到列表中
            correlation_coefficients.append(correlation)
        if np.isnan(np.sum(correlation_coefficients)):
            all_nan = all(np.isnan(correlation_coefficients))
            if all_nan:
                return -10000
            else:
                # 计算非NaN值的平均值
                average = np.mean([x for x in correlation_coefficients if not np.isnan(x)])
                # 用平均数替代缺失值
                correlation_coefficients = [x if not np.isnan(x) else average for x in correlation_coefficients]
        # 计算相关系数的平均值
        average_correlation = np.mean(correlation_coefficients)

        # 计算十分组下的多头组合信息比率
        column_sums = np.sum(factor, axis=0)
        # 计算要保留的列的数量（最大的前10%）
        top_10_percent = int(0.1 * X.shape[1])
        top_column_indices = []
        if average_correlation > 0:
        # 找到最大的前10%列的索引
            top_column_indices = np.argpartition(column_sums, top_10_percent)[:top_10_percent]
        if average_correlation < 0:
        # 找到最小的前10%列的索引
            top_column_indices = np.argpartition(column_sums, -top_10_percent)[-top_10_percent:]
        if average_correlation == 0:
            return -10000
        # 使用索引筛选出十分位的股票组合及其对应的每日收益率
        filtered_y = Y[:, top_column_indices]
        # 读取某一年中证500的日频量价数据文件
        df = pd.read_csv(f"C:/Data/index_data/{year}/zz500.zip", compression='zip', index_col=0)
        df_reverse_sorted = df.iloc[::-1]
        df = df_reverse_sorted.reset_index(drop=True)
        # 给中证500指数每日涨跌幅赋值
        benchmark_returns = df.pct_chg.values / 100
        # 计算超额收益的数据序列（每日）
        excess_returns = filtered_y[2:filtered_y.shape[0], :] - benchmark_returns[2:filtered_y.shape[0], np.newaxis]
        excess_returns = np.mean(excess_returns, axis=1)
        # 计算超额收益的标准差
        tracking_error = np.std(excess_returns)
        # 计算信息比率
        information_ratio = np.mean(excess_returns) / tracking_error
        raw_fitness = information_ratio
        return raw_fitness


    def excess_ret(self, X, Y, year):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_days, n_samples]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Y : array-like, shape = [n_days, n_samples]
            Target values.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        factor = self.execute(X)

        # if self.transformer:
        #     y_pred = self.transformer(y_pred)
        # raw_fitness = self.metric(y, y_pred, sample_weight)
        correlation_coefficients = []
        for i in range(factor.shape[0]-2):
            col_X = factor[i, :]  # 获取X的第i列
            col_Y = Y[i + 2, :]  # 获取Y的第i+2列，因为从第1列开始与第3列对应
            # 计算相关系数
            correlation = np.corrcoef(col_X, col_Y)[0, 1]
            # 将相关系数添加到列表中
            correlation_coefficients.append(correlation)
        if np.isnan(np.sum(correlation_coefficients)):
            all_nan = all(np.isnan(correlation_coefficients))
            if all_nan:
                return -10000
            else:
                # 计算非NaN值的平均值
                average = np.mean([x for x in correlation_coefficients if not np.isnan(x)])
                # 用平均数替代缺失值
                correlation_coefficients = [x if not np.isnan(x) else average for x in correlation_coefficients]
        # 计算相关系数的平均值
        average_correlation = np.mean(correlation_coefficients)
        # 计算十分组下的多头组合信息比率
        column_sums = np.sum(factor, axis=0)
        # 计算要保留的列的数量（最大的前10%）
        top_10_percent = int(0.1 * X.shape[1])
        top_column_indices = []
        if average_correlation > 0:
        # 找到最大的前10%列的索引
            top_column_indices = np.argpartition(column_sums, top_10_percent)[:top_10_percent]
        if average_correlation < 0:
        # 找到最小的前10%列的索引
            top_column_indices = np.argpartition(column_sums, -top_10_percent)[-top_10_percent:]
        if average_correlation == 0:
            return -10000
        # 使用索引筛选出十分位的股票组合及其对应的每日收益率
        filtered_y = Y[:, top_column_indices]
        # 读取中证500的日频量价数据文件
        df = pd.read_csv(f"C:/Data/index_data/{year}/zz500.zip", compression='zip', index_col=0)
        df_reverse_sorted = df.iloc[::-1]
        df = df_reverse_sorted.reset_index(drop=True)
        # 给中证500指数每日涨跌幅赋值
        benchmark_returns = df.pct_chg.values / 100
        # 计算超额收益的数据序列（每日） 并计算年化超额收益
        excess_returns = filtered_y[2:filtered_y.shape[0], :] - benchmark_returns[2:filtered_y.shape[0], np.newaxis]
        excess_returns = np.mean(excess_returns, axis=1)
        year_ret = np.sum(excess_returns)
        return year_ret*100


    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)

        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) - set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:

            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                if replacement.is_ts:
                    replacement = make_ts_function(replacement, self.d_ls, random_state)
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    # def fitness(self, X, Y, parsimony_coefficient=None):
    #     """Evaluate the penalized fitness of the program according to X, y.
    #
    #     Parameters
    #     ----------
    #     parsimony_coefficient : float, optional
    #         If automatic parsimony is being used, the computed value according
    #         to the population. Otherwise the initialized value is used.
    #
    #     Returns
    #     -------
    #     fitness : float
    #         The penalized fitness of the program.
    #
    #     """
    #     if parsimony_coefficient is None:
    #         parsimony_coefficient = self.parsimony_coefficient
    #     penalty = parsimony_coefficient * len(self.program)
    #     fitness = self.raw_fitness(X, Y) - penalty
    #     return fitness


    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)


def make_ts_function(function, d_ls, random_state):
    """
    Parameters
    ----------
    function: _Function
        时间序列函数.

    d_ls: list
        参数 'd' 可选范围.

    random_state: RandomState instance
        随机数生成器.

    """
    d = random_state.randint(len(d_ls))
    d = d_ls[d]
    function_ = copy(function)
    function_.set_d(d)
    return function_




# 后续需要改进：
# fitness函数没有加上惩罚系数