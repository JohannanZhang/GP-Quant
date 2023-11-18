import tushare as ts
import numpy as np
import pandas as pd
import os




path = "C:/Data/stock_data/ts_data/"
pro = ts.pro_api()

"""
======================获得A股的特定时间范围内的交易日历，以及添加周标记的交易日历，并保存在本地文件
"""
# 获得回测时间段内的所有日期以及是否交易属性，添加星期分组后保存
# df = pro.trade_cal(exchange='', start_date='20210101', end_date='20230701')
# df_reverse_sorted = df.iloc[::-1]
# df = df_reverse_sorted.reset_index(drop=True)
# df['cal_date'] = pd.to_datetime(df['cal_date'])
# # 将日期不同的星期进行分组，并打印出每个星期第一个交易日对应的行索引并保存为numpy属性
# df['year_week'] = df['cal_date'].dt.strftime('%Y-%U')
# df.to_csv(path+'trade_date.csv', index=False)






"""
======================一、保存A股所有股票的上市日期、退市股票的退市日期，以及当下暂停上市的股票
"""
# 上市
# list_date = pro.stock_basic(list_status='L', fields='ts_code, list_date')
# list_date['list_date'] = pd.to_datetime(list_date['list_date'])
# print(list_date)
# list_date.to_csv(path + 'list_date.csv', index=False)
# 退市
# delist_date = pro.stock_basic(list_status='D', fields='ts_code, delist_date')
# delist_date['delist_date'] = pd.to_datetime(delist_date['delist_date'])
# print(delist_date)
# delist_date.to_csv(path+'delist_date.csv', index=False)
# 暂停上市
# pause_date = pro.stock_basic(list_status='P')
# pause_date.to_csv(path + 'pause_date.csv', index=False)



"""
======================二、保存标的指数成分股的曾用名文件（基于2020年底调整后，用于排除ST、ST摘帽不到2个月的股票） 注：encoding为‘gbk’
"""
# df = pro.index_weight(index_code='000905.SH', start_date='20201201', end_date='20201231')
# print(df)
# # df.to_csv(path+'stock_list.csv', index=False)
# df = df['con_code']
#
# # 基于2020年底调整后的中证500指数成分股，调用tushare生成各个股票对应的曾用名文件（此处tushare接口每分钟只能调用100只，实际保存时500只股票分批保存）
# stock_list = []
# for i in range(len(df)):
#     stock_list.append(df.iloc[i])
# for j in range(100):
#     code = stock_list[j+400]
#     data = pro.namechange(ts_code=code, fields='ts_code,name,start_date,end_date,ann_date, change_reason')
#     file_name = path+'used_names/'+code[:-3] + '.csv'
#     data.to_csv(file_name, encoding='gbk', index=False)
#     print(f"第{j+401}个元素已完成")





"""
=============================三、查询每只股票的停复牌信息，并保存
"""
df = pro.index_weight(index_code='000905.SH', start_date='20201201', end_date='20201231')
df = df['con_code']

# 基于2020年底调整后的中证500指数成分股，调用tushare生成各个股票对应的曾用名文件（此处tushare接口每分钟只能调用100只，实际保存时500只股票分批保存）
# stock_list = []
# for i in range(len(df)):
#     stock_list.append(df.iloc[i])
# print(stock_list)
#
#
# for j in range(100):
#     code = stock_list[j+400]
#     data = pro.suspend(ts_code=code, suspend_date='', resume_date='', fields='')
#     file_name = path+'suspend_date/'+code[:-3] + '.csv'
#     data.to_csv(file_name, encoding='gbk', index=False)
#     print(f"第{j+401}个元素已完成")








