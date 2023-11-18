# 基于标的资产（中证500指数）过去120个交易日内的8个因子的多头组合信息比率这个指标进行加权，选出接下来一周的持仓，并给予每周的持仓变动，计算交易行为以及最终收益
# 回测时间：2020年初-2023年最新数据

import numpy as np
import pandas as pd
import tushare as ts

pro = ts.pro_api()


filename = 'C:/Data/stock_data/timeshare/2017/002268.zip'
data = pd.read_csv(filename, compression='zip', header=None)
# 给源数据加上列名
column_labels = ['date', 'time', 'open', 'high', 'low', 'close', 'vol', 'turnover']
data.columns = column_labels
# 将date日期类数据类型转为datetime型数据
data['date'] = pd.to_datetime(data['date'])


# selected_rows1 = data[data['date'] == '2017-06-06']
# print(selected_rows1)
# # 涨停当日
# selected_rows2 = data[data['date'] == '2017-06-07']
# print(selected_rows2)


df1 = pro.adj_factor(ts_code='002268.SZ', trade_date='20210806')
print(df1)


df2 = pro.stk_factor(ts_code='002268.SZ', start_date='20080801')
df_reverse_sorted = df2.iloc[::-1]
df = df_reverse_sorted.reset_index(drop=True)
S = df[["ts_code", "trade_date", "adj_factor", "pct_change"]]
print(S)


# df = ts.pro_bar(ts_code='002268.SZ', freq='1MIN', adj='hfq', start_date='20080101', end_date='20181011')
#
# print(df)
# df.to_csv("C:/Users/zhangyuhang/Desktop/e.csv", index=None)




