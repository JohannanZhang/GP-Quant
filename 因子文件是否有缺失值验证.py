import numpy as np
import pandas as pd
import os

factor_data_path = "C:/Data/stock_data/factor/gplearn/virable/2017"


for filename in os.listdir(factor_data_path):
    data = pd.read_csv(f"C:/Data/stock_data/factor/gplearn/virable/2017/{filename}", header=None)
    has_missing_values = data.isna().any().any()
    if has_missing_values:
        print(f"{filename}有缺失值")


file_path = 'C:/Data/stock_data/factor/gplearn/virable/2017/600289.csv'
df = pd.read_csv(file_path, header=None)
# 打印选择的数据
missing_rows = df[df.isna().any(axis=1)].index

# 查找缺失值的列索引
missing_columns = df.columns[df.isna().any()]

# 打印缺失值的行索引和列索引
print("缺失值的行索引：")
print(missing_rows)
print("\n缺失值的列索引：")
print(missing_columns)

