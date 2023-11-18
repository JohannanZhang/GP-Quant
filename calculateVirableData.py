import tushare as ts
import pandas as pd
import numpy as np
import preprocess as pp
import os
import multiprocessing
from functools import partial



# 获取某年1月份的中证500对应的个股代码列表、以及以开盘价为基准的每日涨跌幅
def get_stock_list(year: int):
    pro = ts.pro_api()
    df = pro.index_weight(index_code='000905.SH', start_date=f'{year}0101', end_date=f'{year}0131')
    # 提取con_code列中的6位代码数据并导出为列表
    output_data = []
    for _, row in df.iterrows():
        con_code = row['con_code'][:-3]  # 删除最后三位（".SH"）
        output_data.append(con_code)
    return output_data



# 读取固定股票，固定年份的数据，并计算其对应的供84个因子的值，并进行预处理（去极值，0-1标准化处理），与日频收益率数据一同存储在特定地址
def get_virable_data(year: int, symbol: str):
    # 第一步：导入数据，并删除日成交额低于500万元对应的成交日数据
    df = pp.read_data(year=year, symbol=symbol)
    df = pp.liquid_select(df)
    # 第二步：添加一列0-1标准化的收盘价数据，以及按交易时间、股价高低分出不同的原数据，便于后面求各个因子的值
    df = pp.price_standlize(df)
    df = pp.get_time_range(df)
    df = pp.get_price_range(df)

    df_open = df[df['time_range'] == 'open']
    df_intra = df[df['time_range'] == 'intra']
    df_close = df[df['time_range'] == 'close']

    df_high = df[df['price_range'] == 'high']
    df_low = df[df['price_range'] == 'low']

    # 将数据和函数组合，计算84个变量与日收益率
    dataframes = [df, df_open, df_intra, df_close, df_high, df_low]
    functions = [pp.ret_avg, pp.ret_std, pp.ret_skew, pp.ret_kurt,
                 pp.swap_avg, pp.swap_std, pp.swap_skew, pp.swap_kurt,
                 pp.price_avg, pp.price_std, pp.price_skew, pp.price_kurt,
                 pp.corr_priceswap, pp.corr_deltaswap]
    result_dataframes = []
    for df in dataframes:
        for func in functions:
            result = func(df)
            result_dataframes.append(result.iloc[:, 1])  # 提取第二列数据
    daily_return = pp.daily_return(year=year, symbol=symbol)
    # 计算基于开盘价的日收益率
    result_dataframes.append(daily_return.iloc[:, 1])
    # 将所有结果拼接成一个DataFrame
    final_result = pd.concat(result_dataframes, axis=1)
    result_df = final_result.copy()
    # 更改变量名
    result_df.columns = result_df.columns[:14].tolist() + ['open_' + col for col in result_df.columns[14:28]] + result_df.columns[28:].tolist()
    result_df.columns = result_df.columns[:28].tolist() + ['intra_' + col for col in result_df.columns[28:42]] + result_df.columns[42:].tolist()
    result_df.columns = result_df.columns[:42].tolist() + ['close_' + col for col in result_df.columns[42:56]] + result_df.columns[56:].tolist()
    result_df.columns = result_df.columns[:56].tolist() + ['high_' + col for col in result_df.columns[56:70]] + result_df.columns[70:].tolist()
    result_df.columns = result_df.columns[:70].tolist() + ['low_' + col for col in result_df.columns[70:84]] + result_df.columns[84:].tolist()
    # 对因子进行0-1标准化处理
    for col_name in result_df.columns[:84]:  # 遍历前84列
        result_df = pp.min_max_scaling(result_df, col_name)
    # 删除空缺值所在的行
    result_df = result_df.dropna()
    # 指定保存的文件路径和文件名
    file_path = f'C:/Data/stock_data/factor/gplearn/virable/{year}'
    # # 创建目标文件夹（如果不存在）
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    result_df.to_csv(os.path.join(file_path, f'{symbol}.csv'), index=False)
    # print(result_df)
    # return result_df


# # 读取并存储中证500指数2017年的日频量价数据，以便于后续计算投资组合的超额收益、信息比率
# pro = ts.pro_api()
# df = pro.index_daily(ts_code='399905.SZ', start_date='20170101', end_date='20171231')
# df.to_csv("C:/Data/index_data/2015/zz500.csv")



# 在指定文件夹下计算并生成生成因子数据文件
# def parallel_process(year, symbol_list):
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # 使用所有可用的CPU核心
#     func = partial(get_virable_data, year)
#     pool.map(func, symbol_list)
#     pool.close()
#     pool.join()
# year = 2015
# if __name__ == '__main__':
#     # 调用并行处理函数
#     parallel_process(year, zz500)



# 判断固定年份下的变量文件中是否有缺失值
# import os
# factor_data_path = "C:/Data/stock_data/factor/gplearn/virable/2015"
#
# for filename in os.listdir(factor_data_path):
#     data = pd.read_csv(f"C:/Data/stock_data/factor/gplearn/virable/2015/{filename}", header=None)
#     has_missing_values = data.isna().any().any()
#     if has_missing_values:
#         print(f"{filename}有缺失值")





