import pickle
import numpy as np
import pandas as pd
from readVirableData import read_virable_data
X_2017, Y_2017 = read_virable_data(year=2017)

# 本文件用于对各个子代中最优异的因子集合进行筛选，保留与其他因子相关性小于0.7的因子
X, Y = read_virable_data(year=2017)

all_list = []

# 因子所处的文件
factor_path = "C:/Data/stock_data/factor/gplearn/fuctiontree_save"


# 由运行结果可以得到：适应度（即超额信息比率）超过0.1，且在15年验证表现也良好的子代为第5-100代
better_gen = list(range(4, 100))
# 各代中最优异个体在相应代中的索引：
better_gen_index = [68, 5, 127, 3, 13, 0,
                    0, 0, 0, 1, 2, 46, 34, 116, 8, 2,
                    163, 11, 1, 0, 0, 3, 4, 0, 2, 1,
                    0, 0, 1, 2, 0, 0, 1, 5, 0, 0,
                    0, 3, 7, 2, 0, 2, 1, 2, 0, 0,
                    0, 1, 0, 0, 0, 2, 0, 0, 1, 2,
                    0, 1, 4, 1, 3, 2, 3, 2, 0, 0,
                    0, 0, 0, 0, 2, 2, 0, 1, 0, 0,
                    0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
                    2, 0, 0, 6, 0, 0, 2, 0, 1, 0]

factor_square = np.zeros((0, len(better_gen_index)))

# 计算各个优异子个体（因子）的因子值，以及彼此的相关系数
arrays = []
for i in range(4, 100):
    # 加载保存的文件
    with open(factor_path + "/" + f'{i}_FuncTreeList.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    index = better_gen_index[i-4]
    super_tree = loaded_lists[index]
    factor_square = super_tree.execute(X)
    arrays.append(factor_square)


# 初始化一个空列表，用于存储与其他数组相关性小于0.7的数组
selected_arrays = []
selected_indices = []  # 初始化一个空列表，用于存储对应的数组索引#
# 循环计算相关系数并筛选数组
for i, array1 in enumerate(arrays):
    is_selected = True  # 假设初始状态下数组被选中

    for j, array2 in enumerate(arrays):
        if i != j:  # 确保不与自身比较
            # 计算相关系数
            correlation_coefficients = []
            for k in range(array1.shape[1]):  # 假设这两个数组的列数相同
                row1 = array1[:, k]  # 获取data1的第i行
                row2 = array2[:, k]  # 获取data2的第i行
                # 使用np.corrcoef计算两行之间的相关系数
                correlation_coefficient = np.corrcoef(row1, row2)[0, 1]
                correlation_coefficients.append(correlation_coefficient)
            # 将结果转换为NumPy数组
            correlation_coefficients = np.array(correlation_coefficients)
            correlation_coefficients = np.mean(correlation_coefficients)
            if correlation_coefficients >= 0.7:
                is_selected = False  # 如果有一个相关系数大于等于0.7，将该数组标记为不被选中
                break  # 停止与其他数组的比较

    if is_selected:
        selected_arrays.append(array1)  # 如果与所有其他数组的相关性都小于0.7，将该数组添加到选中列表中
        selected_indices.append(i)


# selected_arrays 现在包含了相关性小于0.7的所有数组
print(selected_arrays)
# 剔除了与其他因子相关系数大于等于0.7的因子后，保存下来的因子对应索引(8个)
print(selected_indices)
print(len(selected_indices))


# 通过以上的代码对96个因子彼此相关系数计算，剔除了与其他因子相关系数大于等于0.7的因子后，保存下来了8个因子
# 这些因子的位置索引为final_index，即以上的计算的selected_indices为：[1, 4, 9, 11, 19, 30, 40, 42]
final_index = selected_indices
# 返回最终8个因子对应的父代索引，以及父代中个体的位置索引
final_gen = [better_gen[i] for i in final_index]
print(final_gen)
final_son_index = [better_gen_index[i] for i in final_index]
print(final_son_index)
super_programs = []


# 最终选择的父代索引final_gen，以及相应个体索引final_son_index分别为：
# final_gen = [5, 8, 13, 15, 23, 34, 44, 46]
# final_son_index = [5, 13, 1, 46, 0, 0, 0, 1]

for i in range(len(final_gen)):
    # 读取每一个合格的父代的索引
    parent_index = final_gen[i]
    son_index = final_son_index[i]
    with open(factor_path + "/" + f'{parent_index}_FuncTreeList.pkl', 'rb') as file:
        loaded_lists = pickle.load(file)
    super_program = loaded_lists[son_index]
    super_programs.append(super_program)


# 现在，将all_lists保存到文件中，以便后续分析
with open('C:/Data/stock_data/factor/gplearn/fuctiontree_save/super_FuncTreeList.pkl', 'wb') as file:
    pickle.dump(super_programs, file)


