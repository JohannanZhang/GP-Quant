import pandas as pd
import numpy as np
from calculateVirableData import get_stock_list
"""
读取基于calculateVirableData存储的，特定年份下的所有股票84列的二维数据，通过剔除缺失行过多股票数据、剩余股票数据缺失行填充的方式，得到维度为（有效交易日数，有效交易股票数，变量数）的三维变量值数组array_x， 
以及维度为（有效交易日数，有效交易股票数）的二维数组array_y，表示每只股票基于开盘价的每日收益率。
array_x将被用于后续同公式树结合，计算因子值。array_y将用于因子的筛选。
"""

# 获取固定年份文件夹下的变量数据，并将变量与收益分别保存为三维数组X，二维数组Y
def read_virable_data(year: int):
    zz500 = get_stock_list(year=year)
    # 假设您的500个CSV文件都存储在一个文件夹中，并且文件名是按顺序的
    # 指定包含CSV文件的文件夹路径
    folder_path = f'C:/Data/stock_data/factor/gplearn/virable/{year}'  # 请替换为实际文件夹路径
    # 原始数据缺失000982的数据，此处为通畅运行暂时进行剔除
    if year == 2015:
        target_element = '000982'
        if target_element in zz500:
            zz500.remove(target_element)

    # 初始化一个空列表，用于存储DataFrame
    dataframes = []
    # 遍历文件夹中的CSV文件并加载为DataFrame
    for i in zz500:
        file_path = f"{folder_path}/{i}.zip"  # 根据文件名规律构建文件路径
        data = pd.read_csv(file_path, compression='zip')
        dataframes.append(data)  # 将DataFrame添加到列表中

    # 第一步：找到最多的行数 n
    n = max(df.shape[0] for df in dataframes)
    # 第二步：筛选行数大于n/2 的DataFrame
    min_required_rows = n * 0.9  # 假设 n 为已知的最多行数
    filtered_dataframes = []  # 用于存储筛选后的DataFrame
    for df in dataframes:
        if df.shape[0] > min_required_rows:
            filtered_dataframes.append(df)

    sample_num = len(filtered_dataframes)

    # 第三步：对剩余的DataFrame进行行填充
    for i, df in enumerate(filtered_dataframes):
        if df.shape[0] < n:
            # 计算每一列的均值
            column_means = df.mean()
            # 创建需要填充的行数
            rows_to_add = n - df.shape[0]
            # 生成包含均值的新行
            new_rows = pd.DataFrame([column_means] * rows_to_add, columns=df.columns)
            # 将新行添加到DataFrame中
            filtered_dataframes[i] = pd.concat([df, new_rows], ignore_index=True)
    # filtered_dataframes为一个列表，元素为个股dataframe数据，行数为最大交易日数n，列数为85（变量数84+每日收益率daily_return）
    # 元素个数为以上筛选后的样本股票数sample_num

    # 创建一个空的三维数组array_x，维度为 (n交易日数, sample_num样本数, 变量数84)，用于保存样本股票当年的变量数据
    shape_x = (n, sample_num, 84)
    array_x = np.empty(shape_x)
    # 使用循环遍历每个DataFrame并将其数据添加到数组中
    for i, df in enumerate(filtered_dataframes):
        # 将DataFrame数据转换为NumPy数组，并存储在相应的位置
        array_x[:, i, :] = df.iloc[:, 0:84].values
    # 创建一个空的二维数组array_y，维度为 (n交易日数, sample_num样本数)，用于保存样本股票当年的每日基于开盘价的收益率
    shape_y = (n, sample_num)
    array_y = np.empty(shape_y)
    # 使用循环遍历每个DataFrame并将其数据添加到数组中
    for i, df in enumerate(filtered_dataframes):
        # 提取每个DataFrame的第85列数组存储在 result_array 的第 i 列
        array_y[:, i] = df.iloc[:, 84].values

    return array_x, array_y



