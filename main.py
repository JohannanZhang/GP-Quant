import numpy as np
from readVirableData import read_virable_data
from GPLEARN import ts_function_set, function_set
from custom_program import _Program
import pickle
from gplearn.utils import _partition_estimators, check_random_state
from joblib import Parallel, delayed
import itertools

MAX_INT = np.iinfo(np.int32).max

X_2015, Y_2015 = read_virable_data(year=2015)
X_2017, Y_2017 = read_virable_data(year=2017)


arities = {
    1: [ts_function_set[0], ts_function_set[1], ts_function_set[2], ts_function_set[3], ts_function_set[4],
        ts_function_set[5], ts_function_set[6], ts_function_set[7], ts_function_set[8], function_set[10],
        function_set[9], function_set[8], function_set[7], function_set[6]],
    2: [function_set[0], function_set[1], function_set[2],
        function_set[3], function_set[4], function_set[5]]}


random_state = np.random.RandomState()
parents = []
tournament_size = 20
population_size = 200
generations = 100
n_jobs = 12


for i in range(population_size):
    parent = _Program(ts_function_set=ts_function_set,
                    random_state=random_state,
                    arities=arities,
                    d_ls=[4, 5, 6, 7, 8, 9, 10],
                    function_set=function_set,
                    init_depth=(3, 5),
                    init_method='half and half',
                    n_features=84,
                    parsimony_coefficient=0.1,
                    const_range=None,
                    p_point_replace=0.40,
                    transformer=None,
                    feature_names=None,
                    program=None)
    parents.append(parent)


# 在父代的基础上，通过锦标赛筛选法，选出规模为n_programs的子代列表
# 基于固定年份year的变量X、收益Y，以及当年的基准收益，计算个体筛选的适应度(即对应的当年超额信息比率)
def parallel_evolve(n_programs, parents, X, Y, seeds, year: int):
    def tournament():
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].raw_fitness(X, Y, year=year) for p in contenders]
        parent_index = contenders[np.argmax(fitness)]
        # print("优秀的父代个体：", parents[parent_index], "优秀个体对应的索引", parent_index, "优秀个体对应的适应度：", fitness[np.argmax(fitness)])
        return parents[parent_index], parent_index
    programs = []
    # 下一代：基于父代列表parents，循环结构选择出相等于父代数目的优秀子代，并进行突变，
    for i in range(n_programs):
        random_state = check_random_state(seeds[i])
        method = random_state.uniform()
        parent, parent_index = tournament()
        if 0 < method < 0.3:
            # crossover
            donor, donor_index = tournament()
            program, removed, remains = parent.crossover(donor.program,
                                                         random_state)
            genome = {'method': 'Crossover',
                      'parent_idx': parent_index,
                      'parent_nodes': removed,
                      'donor_idx': donor_index,
                      'donor_nodes': remains}
        elif 0.3 <= method < 0.4:
            # subtree_mutation
            program, removed, _ = parent.subtree_mutation(random_state)
            genome = {'method': 'Subtree Mutation',
                      'parent_idx': parent_index,
                      'parent_nodes': removed}
        elif 0.40 <= method < 0.45:
            # hoist_mutation
            program, removed = parent.hoist_mutation(random_state)
            genome = {'method': 'Hoist Mutation',
                      'parent_idx': parent_index,
                      'parent_nodes': removed}
        elif 0.45 <= method < 0.55:
            # point_mutation
            program, mutated = parent.point_mutation(random_state)
            genome = {'method': 'Point Mutation',
                      'parent_idx': parent_index,
                      'parent_nodes': mutated}
        else:
            # reproduction
            program = parent.reproduce()
            genome = {'method': 'Reproduction',
                      'parent_idx': parent_index,
                      'parent_nodes': []}

        son = _Program(ts_function_set=ts_function_set,
                            random_state=random_state,
                            arities=arities,
                            d_ls=[3, 4, 5, 6, 7, 8, 9, 10],
                            function_set=function_set,
                            init_depth=(3, 5),
                            init_method='full',
                            n_features=84,
                            parsimony_coefficient=0.1,
                            const_range=None,
                            p_point_replace=0.40,
                            transformer=None,
                            feature_names=None,
                            program=program)

        # 输出每一个子代program对象，并保存其突变信息，再将所有子代保存在programes列表中
        son.parents = genome
        programs.append(son)
    return programs


population = [[] for _ in range(generations)]
better_gen = []
better_gen_index = []
for gen in range(generations):
    if gen == 0:
        Parents = parents
    else:
        Parents = population[gen - 1]

    n_jobs, n_programs, starts = _partition_estimators(population_size, n_jobs=n_jobs)
    seeds = random_state.randint(MAX_INT, size=population_size)

    Program = Parallel(n_jobs=n_jobs, verbose=5)(
    delayed(parallel_evolve)(n_programs[i],
                              Parents,
                              X_2017,
                              Y_2017,
                              seeds[starts[i]:starts[i + 1]],
                              year=2017,)
    for i in range(n_jobs))
    # Reduce, maintaining order across different n_jobs
    Programs = list(itertools.chain.from_iterable(Program))
    population[gen] = Programs
    # 将第gen代的公式树列表保存在相应文件夹下
    with open(f'C:/Data/stock_data/factor/gplearn/fuctiontree_save/{gen}_FuncTreeList.pkl', 'wb') as file:
        pickle.dump(Programs, file)

    fit_list = []
    for j in population[gen]:
        try:
            fit = j.raw_fitness(X_2017, Y_2017, year=2017)
            fit_list.append(fit)
        except Exception as e:
            print(f"第{gen + 1}代,索引为{j}的公式树出现问题")

    # 取这一代适应度最优个体的索引，及其对应的_Program对象
    fittest_index = fit_list.index(max(fit_list))
    better_gen_index.append(fittest_index)
    fittest = population[gen][fittest_index]

    fitness_2017 = max(fit_list)
    excess_ret_2017 = fittest.excess_ret(X_2017, Y_2017, year=2017)
    print(f"第{gen + 1}代的公式树列表为", Programs)
    print(f"第{gen + 1}代最优个体的位置为：{fittest_index}  第{gen + 1}代最优个体的的适应度：{max(fit_list)}  ")
    print(f"第{gen + 1}代的年化超额收益为：{excess_ret_2017}")

    if fitness_2017 > 0 and excess_ret_2017 > 0:
        fitness_2015 = fittest.raw_fitness(X_2015, Y_2015, year=2015)
        excess_ret_2015 = fittest.excess_ret(X_2015, Y_2015, year=2015)
        if fitness_2015 >= 0.7*fitness_2017:
            print(f'该因子通过测试集，15年信息比率为{fitness_2015}，达到17年的70%')
            better_gen.append(gen)
        if excess_ret_2015 >= 0.7 * excess_ret_2017:
            print(f'该因子通过测试集，15年年化超额收益为{excess_ret_2015}，达到17年的70%')
            better_gen.append(gen)
        else:
            print("该因子并未通过测试集")

better_gen = list(set(better_gen))
print("每一代最优个体的索引列表为：", better_gen_index)
print("通过测试集的因子代数列表为：", better_gen)




