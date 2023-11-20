"""
读取由factor_select得到的最终有效的公式树因子文件super_FuncTreeList.pkl，打印出每一个因子信息（公式树结构、IC、IR，以及于训练集17年，测试集15年的选股基于中证500的超额收益），
并作图表现出各个因子在测试集年份中选股组合的累计收益、基准指数收益，以及超额收益。
"""
import pandas as pd
import numpy as np
import pickle
from readVirableData import read_virable_data
import matplotlib.pyplot as plt
# 选择与pyplot兼容的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

factor_path = "C:/Data/stock_data/factor/gplearn/fuctiontree_save"
X_2015, Y_2015 = read_virable_data(year=2015)
X_2017, Y_2017 = read_virable_data(year=2017)

# # # 读取由最优的8个因子对应的Program对象组成的列表
with open(factor_path + "/" + 'super_FuncTreeList.pkl', 'rb') as file:
    loaded_lists = pickle.load(file)


file_path = "C:/Data/stock_data/factor/gplearn/virable/2015/600062.zip"  # 根据文件名规律构建文件路径
data = pd.read_csv(file_path, compression='zip')
columns = data.columns

IC = []
IR = []

for i in range(len(loaded_lists)):
    fac = loaded_lists[i].program
    func = []
    a = 0
    for j in fac:
        if isinstance(j, int):
            a = columns[j]
        elif j.is_ts == True:
            a = j.name+'('+f'{j.d}'+')'
        else:
            a = j.name
        func.append(a)
    print(f"因子{i+1}的结构为:", func)

    factor = loaded_lists[i].execute(X_2017)
    correlation_coefficients = []
    for j in range(factor.shape[0] - 2):
        col_X = factor[j, :]  # 获取X的第i列
        # t日结束计算因子值，t+1进行交易，t+2得到收益率，所以关注Y的第i+2行与factor第i行的相关性
        col_Y = Y_2017[j + 2, :]
        # 计算相关系数
        correlation = np.corrcoef(col_X, col_Y)[0, 1]
        # 将相关系数添加到列表中
        correlation_coefficients.append(correlation)
    if np.isnan(np.sum(correlation_coefficients)):
        all_nan = all(np.isnan(correlation_coefficients))
        # 计算非NaN值的平均值
        average = np.mean([x for x in correlation_coefficients if not np.isnan(x)])
        # 用平均数替代缺失值
        correlation_coefficients = [x if not np.isnan(x) else average for x in correlation_coefficients]
    # 计算相关系数的平均值
    average_correlation = np.mean(correlation_coefficients)
    IC.append(average_correlation)
    print(f"因子{i+1}的日均IC为:", average_correlation)

    IR.append(loaded_lists[i].raw_fitness(X_2017, Y_2017, year=2017))
    print(f"基于因子{i+1}的十分组股票超额收益信息比率为:", IR[-1])

    excess_ret_2015 = loaded_lists[i].excess_ret(X_2015, Y_2015, year=2015)
    excess_ret_2017 = loaded_lists[i].excess_ret(X_2017, Y_2017, year=2017)
    print(f"基于因子{i+1}的十分组股票组合于训练集2017年相对中证500的超额收益为:", excess_ret_2017)
    print(f"基于因子{i+1}的十分组股票组合于测试集2015年相对中证500的超额收益为:", excess_ret_2015)

print("因子日均IC列表：", IC)
print("因子信息比率IR列表：", IR)


for i in range(len(loaded_lists)):
    factor = loaded_lists[i].execute(X_2015)
    correlation_coefficients = []
    for j in range(factor.shape[0] - 2):
        col_X = factor[j, :]  # 获取X的第i列
        col_Y = Y_2015[j + 2, :]  # 获取Y的第i+2列，因为从第1列开始与第3列对应
        # 计算相关系数
        correlation = np.corrcoef(col_X, col_Y)[0, 1]
        # 将相关系数添加到列表中
        correlation_coefficients.append(correlation)
    if np.isnan(np.sum(correlation_coefficients)):
        # 计算非NaN值的平均值
        average = np.mean([x for x in correlation_coefficients if not np.isnan(x)])
        # 用平均数替代缺失值
        correlation_coefficients = [x if not np.isnan(x) else average for x in correlation_coefficients]
    # 计算相关系数的平均值
    average_correlation = np.mean(correlation_coefficients)
    # 计算十分组下的多头组合信息比率
    column_sums = np.sum(factor, axis=0)
    # 计算要保留的列的数量（最大的前10%）
    top_10_percent = int(0.1 * X_2015.shape[1])
    top_column_indices = []
    if average_correlation > 0:
        # 找到最大的前10%列的索引
        top_column_indices = np.argpartition(column_sums, top_10_percent)[:top_10_percent]
    if average_correlation < 0:
        # 找到最小的前10%列的索引
        top_column_indices = np.argpartition(column_sums, -top_10_percent)[-top_10_percent:]
    # 使用索引筛选出十分位的股票组合及其对应的每日收益率，组成dataframe数据
    filtered_y = Y_2015[:, top_column_indices]
    # 读取中证500的日频量价数据文件
    df = pd.read_csv(f"C:/Data/index_data/{2015}/zz500.zip", compression='zip', index_col=0)
    df_reverse_sorted = df.iloc[::-1]
    df = df_reverse_sorted.reset_index(drop=True)
    # 给中证500指数每日涨跌幅赋值
    benchmark_returns = df.pct_chg.values / 100
    # 计算超额收益的数据序列（每日） 并计算年化超额收益
    # 在因子与收益率的相关性测试时，是计算的T日因子与T+2收益率数据，所以次数也当年的T+2的收益率开始计算
    # 这里设置对十分组中前10%的股票进行等金额买入，所以每日收益为个股每日收益的均值
    portfolio_ret = np.mean(filtered_y[2:], axis=1)
    excess_returns = portfolio_ret - benchmark_returns[2:]

    port_ret = np.cumprod(1 + np.array(portfolio_ret)) - 1
    exe_ret = np.cumprod(1 + np.array(excess_returns)) - 1
    ref_ret = np.cumprod(1 + np.array(benchmark_returns[2:])) - 1

    # 绘制图表
    plt.figure(figsize=(10, 6))
    # 投资组合超额收益
    plt.plot(exe_ret, label='组合相对基准强弱', color='orange')
    # 投资组合绝对收益
    plt.plot(port_ret, label='单因子增强组合', color='red')
    # 指数基准收益
    plt.plot(ref_ret, label='中证500指数', color='black')
    # 设置图表标题和标签
    plt.title(f'因子{i+1}选股于测试集2015年的表现')
    plt.xlabel('交易日')
    plt.ylabel('收益率')
    # 显示图例
    plt.legend()
    # 显示网格线
    plt.grid(True)
    # 保存图像至指定地址
    # save_path = 'C:/Pycharmproject/My_Work/GPLEARN/单因子增强组合于2015年测试集的收益图像/'
    # file_name = f'factor{i+1}.png'
    # plt.savefig(save_path + file_name)
    # 显示图表
    plt.show()

