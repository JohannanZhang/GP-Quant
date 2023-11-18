import pandas as pd
import numpy as np

import pickle

import pickle

# 初始化一个空列表，用于存储每次循环生成的实例对象组成的列表
all_lists = []

# # 循环开始
# for i in range(10):  # 假设你有10次循环
#     # 在每次循环中生成一个实例对象组成的列表，这里用一个示例列表代替
#     instance_list = [i, i * 2, i * 3]  # 这里可以根据你的实际情况修改
#
#     # 将生成的列表添加到all_lists中
#     all_lists.append(instance_list)
#
#     # 在这里进行你的运算和操作，如果出现错误，你可以在这里捕获异常
#
# # 循环结束
#
# # 现在，将all_lists保存到文件中，以便后续分析
# with open('instance_lists.pkl', 'wb') as file:
#     pickle.dump(all_lists, file)

# 加载保存的文件
with open('C:/Data/stock_data/factor/gplearn/fuctiontree_save/super_programs.pkl', 'rb') as file:
    loaded_lists = pickle.load(file)

# 现在loaded_lists包含了之前保存的所有实例对象组成的列表
# 你可以遍历loaded_lists并进行分析
# for instance_list in loaded_lists:
#     # 在这里进行分析操作，查找问题
#     print(instance_list)
# print(loaded_lists)
# a = loaded_lists[0]
# print(a)
# print(a.program)

for i in range(len(loaded_lists)):
    print(loaded_lists[i])































