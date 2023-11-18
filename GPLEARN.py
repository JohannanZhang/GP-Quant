"""
gplearn代码示例
"""
import gplearn.functions
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz
import math
from custom_program import _Program
from custom_genetic import SymbolicTransformer
from gplearn.genetic import SymbolicTransformer

from custom_function import make_function
from custom_function import _Function

import copy



# 时序算子，输入变量为一维度数和与参数d，输出为最小最大标准化处理后的一维数组)
def _ts_min(x, d):
    result = pd.Series(x).rolling(d, min_periods=1).min()
    result = result.values
    return result


def _ts_max(x, d):
    result = pd.Series(x).rolling(d, min_periods=1).max()
    result = result.values
    return result


def _ts_median(x, d):
    result = pd.Series(x).rolling(d, min_periods=1).quantile(0.5)
    result = result.values
    return result


def _ts_mean(x, d):
    result = pd.Series(x).rolling(d, min_periods=1).mean()
    result = result.values
    return result


# 最小窗口设为1，首位缺失值由其余所有数据的均值填充
def _ts_std(x, d):
    result = pd.Series(x).rolling(d, min_periods=1).std()
    result = result.values
    result[0] = np.mean(result[1:len(result)])
    return result


def _ts_stable(x, d):
    mean = pd.Series(x).rolling(d, min_periods=1).mean()
    mean = mean.values
    std = pd.Series(x).rolling(d, min_periods=1).std()
    std = std.values
    std[0] = np.mean(std[1:len(std)])
    # 查找滚动标准差std数组中为0的元素，并替换为其余非0元素的平均值。平滑处理以避免后续计算mean/std时除数为0的情况
    non_zero_elements = std[std != 0]
    non_zero_mean = np.mean(non_zero_elements)
    std[std == 0] = non_zero_mean
    result = mean / std
    return result


def _ts_minmaxnorm(x, d):
    max = pd.Series(x).rolling(d, min_periods=1).min()
    max = max.values
    min = pd.Series(x).rolling(d, min_periods=1).max()
    min = min.values
    result = np.empty_like(x)
    for i in range(len(x)):
        # 判断max - min是否为零
        if max[i] == min[i]:
            result[i] = 0.5
        else:
            result[i] = (x[i] - min[i]) / (max[i] - min[i])
    return result


def _ts_meanstdnorm(x, d):
    mean = pd.Series(x).rolling(d, min_periods=1).mean()
    mean = mean.values
    std = pd.Series(x).rolling(d, min_periods=1).std()
    std = std.values
    std[0] = np.mean(std[1:len(std)])
    # 查找滚动标准差std数组中为0的元素，并替换为其余非0元素的平均值。平滑处理以避免后续计算mean/std时除数为0的情况
    # 找出非0元素，计算均值后将0元素予以替换
    non_zero_elements = std[std != 0]
    non_zero_mean = np.mean(non_zero_elements)
    std[std == 0] = non_zero_mean
    result = (x-mean) / std
    return result


def _ts_delta(x, d):
    mean = pd.Series(x).rolling(d, min_periods=1).mean()
    mean = mean.values
    result = x - mean
    return result


# def _ts_corr(x, y, d):
#     result = np.empty_like(x, dtype=float)
#     for i in range(1, len(x)):
#         if i < d - 1:
#             x_window = x[0:i+1]
#             y_window = y[0:i+1]
#         else:
#             x_window = x[i - d + 1:i + 1]
#             y_window = y[i - d + 1:i + 1]
#         correlation = np.corrcoef(x_window, y_window)[0, 1]
#         result[i] = correlation
#         result[0] = np.mean(result[1:len(result)])
#     return result



# 目标算子有三大类：①截面多变量算子（6个） ②截面单变量算子（5个） ③时序算子（10个）
# gplearn默认提供了所有的截面多变量算子和部分截面单变量算子，但并未提供时序算子。
# 为了算子的名称格式统一，以下采用对全部算子进行自定义，并导入模型的方式
# 截面多变量算子：
def _cs_add(a, b):
    return a+b


def _cs_sub(a, b):
    return a-b


def _cs_mul(a, b):
    return a*b


def _cs_div(a, b):
    if abs(b) < 0.001:
        return 1
    else:
        return a/b


def _cs_min(a, b):
    return np.minimum(a, b)


def _cs_max(a, b):
    return np.maximum(a, b)


# 截面单变量算子
def _cs_sqrt(a):
    return abs(a) ** (1/2)


def _cs_curt(a):
    return np.cbrt(a)


def _cs_square(a):
    return a ** 2


def _cs_cube(a):
    return a ** 3


def _cs_log(a):
    if abs(a) < 0.001:
        return 0
    else:
        return math.log(abs(a))



# 一、自定义运算符号（custom functions）
# functions.make_function()
# 注意：①返回值为形状符合要求的numpy数组 ②函数要有一些检查，不能出现除数为0和无效运算（如log -2）
cs_add = _Function(function=_cs_add, name="cs_add ", arity=2)
cs_sub = _Function(function=_cs_sub, name="cs_sub ", arity=2)
cs_mul = _Function(function=_cs_mul, name="cs_mul ", arity=2)
cs_div = _Function(function=_cs_div, name="cs_div ", arity=2)
cs_max = _Function(function=_cs_max, name="cs_max ", arity=2)
cs_min = _Function(function=_cs_min, name="cs_min ", arity=2)
cs_sqrt = _Function(function=_cs_sqrt, name="cs_sqrt ", arity=1)
cs_curt = _Function(function=_cs_curt, name="cs_curt ", arity=1)
cs_square = _Function(function=_cs_square, name="cs_square ", arity=1)
cs_cube = _Function(function=_cs_cube, name="cs_cube ", arity=1)
cs_log = _Function(function=_cs_log, name="cs_log ", arity=1)

ts_max = _Function(function=_ts_max, name='ts_max', arity=1, is_ts=True)
ts_min = _Function(function=_ts_min, name='ts_min', arity=1, is_ts=True)
ts_median = _Function(function=_ts_median, name='ts_median', arity=1, is_ts=True)
ts_mean = _Function(function=_ts_mean, name='ts_mean', arity=1, is_ts=True)
ts_std = _Function(function=_ts_std, name='ts_std', arity=1, is_ts=True)
ts_stable = _Function(function=_ts_stable, name='ts_stable', arity=1, is_ts=True)
ts_minmaxnorm = _Function(function=_ts_minmaxnorm, name='ts_minmaxnorm', arity=1, is_ts=True)
ts_meanstdnorm = _Function(function=_ts_meanstdnorm, name='ts_meanstdnorm', arity=1, is_ts=True)
ts_delta = _Function(function=_ts_delta, name='ts_delta', arity=1, is_ts=True)
# ts_corr = _Function(function=_ts_corr, name='ts_corr', arity=2, is_ts=True)


# gp = SymbolicTransformer(function_set=[cs_add, cs_sub, cs_mul, cs_div, cs_max, cs_min, cs_sqrt, cs_curt,
#                                        cs_cube, cs_log, ts_max, ts_min, ts_median,
#                                        ts_mean, ts_std, ts_stable, ts_minmaxnorm,
#                                        ts_meanstdnorm, ts_delta, ts_corr])



# 2017年训练集数据 X（244，488，84） Y（244，488）
# import datafetch
# X=datafetch.array_x
# Y=datafetch.array_y



ts_function_set = [ts_max, ts_min, ts_median,ts_mean, ts_std, ts_stable,
                   ts_minmaxnorm, ts_meanstdnorm, ts_delta]
# ts_corr
function_set = [cs_add, cs_sub, cs_mul, cs_div, cs_max, cs_min, cs_sqrt, cs_curt, cs_square, cs_cube, cs_log]
# d_ls = [3,4,5,6,7,8,9,10]
# for i in function_set:
#     print(i.name)
# # arities,
# init_depth = 4
# init_method = 'half and half'
# n_features = 84
# const_range = None
#  # metric,
# p_point_replace = 0.40
# parsimony_coefficient = 0.2
# random_state = 0.5
# transformer = None
# feature_names = None
# program = None


# gp1 = SymbolicTransformer(
# generations=10,
# population_size=1000,
# function_set=function_set,
# init_depth=(1,4),
# tournament_size=20,
# metric='spearman',
# p_crossover=0.4,
# p_subtree_mutation=0.01,
# p_hoist_mutation=0,
# p_point_mutation=0.01,
# p_point_replace=0.40,
# warm_start=False,
# verbose=1,
# random_state=0,
# n_jobs=-1,
# feature_names=['open', 'close', 'high', 'low', 'volume', 'return_rate', 'vwap'])
#
# # gp1.fit(train,label)# 训练模型







