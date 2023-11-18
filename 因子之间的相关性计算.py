import pandas as pd
import numpy as np
import random


factorA_transposed = np.transpose(factorA)
factorB_transposed = np.transpose(factorB)

# 初始化一个空的列表，用于存储相关系数
correlation_coefficients = []

# 计算每一列之间的相关系数
for colA, colB in zip(factorA_transposed, factorB_transposed):
    correlation_coefficient = np.corrcoef(colA, colB)[0, 1]
    correlation_coefficients.append(correlation_coefficient)

# 将结果转换为NumPy数组
correlation_coefficients = np.array(correlation_coefficients)

# 打印包含488个相关系数的一维数组
print(correlation_coefficients)
print(len(correlation_coefficients))
print(np.mean(correlation_coefficients))



