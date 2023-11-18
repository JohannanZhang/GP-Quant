import tushare as ts


pro = ts.pro_api()
# df = pro.index_weight(index_code='000905.SH', start_date=f'{2022}0101', end_date=f'{2022}0131')
# # 提取con_code列中的6位代码数据并导出为列表
# output_data = []
# for _, row in df.iterrows():
#     con_code = row['con_code'][:-3]  # 删除最后三位（".SH"）
#     output_data.append(con_code)
#
# print(output_data)
# print(len(output_data))




"""
现在开始模拟在2022年1月4日（第一个交易日）时，第一次建仓的股票池选取
"""

# 先行判断中证500的成分股有哪些，这种判断取决于当下选股的时间点对应的中证500成分股有哪些
# # 上证50、沪深300、中证500每年的6月、12月份进行权重股调整
# # 中证500的调整是在每年12月第二个星期的下一交易日
# 此处为：
# 利用tushare模块导出该时间段对应的中证500指数，并将股票代码导出为列表
df1 = pro.index_weight(index_code='000905.SH', start_date='20221201', end_date='20221230')
print(df1)

output_data = []
for _, row in df1.iterrows():
    con_code = row['con_code'][:-3]  # 删除最后三位（".SH"）
    output_data.append(con_code)
print(output_data)
print(len(output_data))




# 下面处理第一个问题：
# 剔除掉上市半年以内的股票







