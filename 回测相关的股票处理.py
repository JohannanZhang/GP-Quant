import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 基于短周期量价回测的框架大致如下（日后边做边更改）
# ①读取回测期间的每一个交易日（交易日历），找出做调仓的那些交易日（此处为周频调仓）
# ②基于每个调仓的日期做判断，筛选出目标股票池
# ③结合目标股票池以及相应因子的取值，计算各个因子的IC、IR，最终得到基于加权因子的选股组合
# ④将下一期的目标组合与当下持仓做对比，进行调仓（此处还要考虑实际停牌、涨跌停等问题，是否能调仓）
# ⑤记录所做的交易与新的仓位情况，以及当期、总损益


path = 'C:/Data/stock_data/ts_data/'
# 一、读取回测期间的每一个交易日（交易日历），找出做调仓的那些交易日（此处为周频调仓）
pro = ts.pro_api()

# ①读取交易日历，找出调仓日期，回测区间为2021年年初，直至2023年7月
# 读取保存的回测时间段内交易日历文件，保存每星期第一个交易日与对应的索引
df = pd.read_csv(path+'trade_date.csv')
first_open_indices = df[df['is_open'] == 1].groupby('year_week')['cal_date'].idxmin()
selected_indexes = first_open_indices.values
selected_rows = df.iloc[selected_indexes]
trade_days = selected_rows['cal_date'].tolist()
print(trade_days)


# 二、基于每个调仓的日期做判断，筛选出目标股票池
# 首先进行对标的股票总体的预处理：
# 1、新股问题：不少股票上市时间早于回测开始的半年前，不用在此后每周做重复判断，
# 2、ST股票：多股票从没有被列入过ST，也可以先一步筛除（此处是为减少程序运算，并未运用到未来函数，每个交易日依旧会对股票名称是否挂ST做判断）
# 3、筛选出在回测时间段内退市的股票，方便后面的回测筛选


# 新股问题预筛选
# stock_list = pd.read_csv(path+'stock_list.csv')
# print(stock_list)
# list_date = pd.read_csv(path+'list_date.csv')
# print(list_date)
# result_df = stock_list.merge(list_date, left_on='con_code', right_on='ts_code', how='left')
# print(result_df)
# half_year_before = pd.to_datetime(trade_days[0]) - pd.DateOffset(months=6)
# result_df['list_date'] = pd.to_datetime(result_df['list_date'])
# condition = result_df['list_date'] > half_year_before
# # 筛选出此后需要判断的股票信息
# list_judge = result_df[condition]


# 接着是历史上有挂ST的股票筛选：
# df = pd.read_csv(path+'used_names/000008.csv', encoding='gbk')
# print(df)
# a = df['name'].str.contains('ST')
# print(a)
# print(type(a))
# 读取所有目标股票的曾用名文件，如果有
# file_names = os.listdir(path+'used_names')
# df_list = []
# for i in file_names:
#     data = pd.read_csv(path+f'used_names/{i}', encoding='gbk')
#     df_list.append(data)
# print(df_list)
#
#
# st_judge_list = []
# for j in range(len(df_list)):
#     s = df_list[j]['name'].str.contains('ST')
#     if s.any():
#         code = df_list[j]['ts_code'].iloc[0]
#         st_judge_list.append(code)
# print(st_judge_list)
# print(len(st_judge_list))


# 筛选出标的股票中有退市的股票及其退市时间
# stock_list = pd.read_csv(path+'stock_list.csv')
# delist_date = pd.read_csv(path+'delist_date.csv')
# result_df = stock_list.merge(delist_date, left_on='con_code', right_on='ts_code', how='left')
# df_filtered = result_df.dropna(subset=['ts_code', 'delist_date'])
# df_filtered.reset_index(drop=True, inplace=True)
# print(df_filtered)


# 预筛选出在回测期间有停牌情况的股票
# 查询单只股票的停复牌信息
df = pd.read_csv('C:/Data/stock_data/ts_data/suspend_date/000031.csv', encoding='gbk')
nearest_date = df['suspend_date'].iloc[0]
# nearest_date = pd.to_datetime(nearest_date)
date = datetime.strptime(nearest_date, "%Y%m%d")

print(date)





"""
==========================关于追踪的指数成分股读取问题============================
"""
# # 上证50、沪深300、中证500每年的6月、12月份进行权重股调整
# # 中证500的调整是在每年12月第二个星期的下一交易日
# df1 = pro.index_weight(index_code='000905.SH', start_date='20170601', end_date='20170630')
# df2 = pro.index_weight(index_code='000905.SH', start_date='20171201', end_date='20171230')
# # 这两个命令反映的是成分股调整月的月末指数成分,用于接下来的半年筛选


"""
===========================关于股票上市日期查询、以及是否与建仓时间相距半年的判断============================
"""
# data = pro.stock_basic(ts_code='002268.SZ', fields='list_date')
# print(data)
#
# data['list_date'] = pd.to_datetime(data['list_date'])
# print(data)
# # print(data.loc[0, 'list_date'])
#
#
# date2 = datetime.strptime('2022-01-04', '%Y-%m-%d')
# #
# time_delta = date2 - data.loc[0, 'list_date']
#
# print(time_delta.days)
#
# half_year = 182.5
# print(time_delta.days < half_year)




# data = pro.stock_basic(fields='ts_code, list_date')
# print(data)





# 将股票代码依据所属市场添加“.SH”或者“.SZ”
# processed_list = []
#
# # 遍历原始列表中的每个元素
# for item in original_list:
#     # 根据元素的开头字符来添加后缀
#     if item.startswith('6'):
#         processed_list.append(item + '.SH')
#     elif item.startswith('3') or item.startswith('0'):
#         processed_list.append(item + '.SZ')
#
# # 打印处理后的列表
# print(processed_list)















