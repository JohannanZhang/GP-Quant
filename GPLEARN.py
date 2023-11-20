"""
①定义目标函数（6个截面多变量函数、5个截面单变量函数、9个时序函数）
②将目标函数实例化为算子的类对象
"""
import numpy as np
import pandas as pd
import math
from custom_function import _Function


# 目标算子有三大类：①截面多变量算子（6个） ②截面单变量算子（5个） ③时序算子（9个）
# 为了算子的名称格式统一，以下先对对全部函数进行自定义，再创建相应的算子类对象。
# 截面多变量函数：
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


# 截面单变量函数
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


# 时序函数，输入变量为一维数组与参数d，输出为最小最大标准化处理后的一维数组)
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


# 定义以上函数对应的算子类对象
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


ts_function_set = [ts_max, ts_min, ts_median,ts_mean, ts_std, ts_stable,
                   ts_minmaxnorm, ts_meanstdnorm, ts_delta]
function_set = [cs_add, cs_sub, cs_mul, cs_div, cs_max, cs_min, cs_sqrt, cs_curt, cs_square, cs_cube, cs_log]







