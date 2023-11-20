import pandas as pd
import numpy as np
import tushare as ts
from scipy.stats import kurtosis


DATA_PATH = 'C:/Data/stock_data/timeshare'

pro = ts.pro_api()
# 读取特定年份下特定股票的分时量价数据，并对价格进行前复权。之后加入分时换手率、收益率数据
def read_data(year: int, symbol: str):
    filename = f'C:/Data/stock_data/timeshare/{year}/{symbol}.zip'
    data = pd.read_csv(filename, compression='zip', header=None)
    # 给源数据加上列名
    column_labels = ['date', 'time', 'open', 'high', 'low', 'close', 'vol', 'turnover']
    data.columns = column_labels
    # 将date日期类数据类型转为datetime型数据
    data['date'] = pd.to_datetime(data['date'])
    # 复权：使用tushare的复权因子数据，对股价进行前复权处理，即：复权后股价(开高低收) = 当日股价（开高低收） × 当日复权因子 / 当年最新复权因子
    ts_symbol = ''
    if symbol.startswith("6"):
        ts_symbol = symbol + ".SH"
    elif symbol.startswith("3") or symbol.startswith("0"):
        ts_symbol = symbol + ".SZ"
    ts_dfA = pro.adj_factor(ts_code=ts_symbol, start_date=f'{year}0101')
    ts_dfA = ts_dfA.rename(columns={'trade_date': 'date'})
    ts_dfA['date'] = pd.to_datetime(ts_dfA['date'])
    ts_dfA['adj_factor'] = ts_dfA['adj_factor'] / ts_dfA['adj_factor'][0]
    merged_df = data.merge(ts_dfA, on='date')
    merged_df['open'] = merged_df['open']*merged_df['adj_factor']
    merged_df['high'] = merged_df['high']*merged_df['adj_factor']
    merged_df['low'] = merged_df['low']*merged_df['adj_factor']
    merged_df['close'] = merged_df['close']*merged_df['adj_factor']
    merged_df.drop('adj_factor', axis=1, inplace=True)
    merged_df.drop('ts_code', axis=1, inplace=True)

    # 计算分钟级的收益率
    grouped = merged_df.groupby('date')
    merged_df['minute_return'] = grouped['close'].pct_change() * 100
    merged_df['minute_return'].fillna(0, inplace=True)

    # 加入分时换手率数据
    ts_dfB = pro.daily_basic(ts_code=ts_symbol, start_date=f'{year}0101', end_date=f'{year}1231',
                             fields='trade_date,turnover_rate_f')
    ts_dfB = ts_dfB.rename(columns={'trade_date': 'date'})
    ts_dfB['date'] = pd.to_datetime(ts_dfB['date'])
    merged_df = merged_df.merge(ts_dfB, on='date')
    grouped = merged_df.groupby('date')
    # 计算每个交易日的总成交量
    total_volume = grouped['vol'].transform('sum')
    # 计算每天的分时换手率
    merged_df['minute_swap_rate'] = (merged_df['vol'] / total_volume) * merged_df['turnover_rate_f']
    merged_df.drop('turnover_rate_f', axis=1, inplace=True)

    return merged_df


# 日内分钟线数据预处理
# 对日期分组并计算每个组的最大值和最小值,对每个交易日的close列进行最大最小标准化
def price_standlize(df: pd.DataFrame):
    grouped = df.groupby('date')
    max_high = grouped['high'].transform('max')
    min_low = grouped['low'].transform('min')
    df['stand_close'] = (df['close'] - min_low) / ((max_high - min_low) + 0.0001)
    return df


# 日内分钟数据基础降频操作（定义日频因子的计算函数）
# 缺失值处理：①单一数值序列，峰度为nan的情况：填充峰度序列的平均值 ②2个相关系数的缺失值情况，填充对应相关系数序列的平均值
# 收益率类
def ret_avg(df: pd.DataFrame):
    grouped = df.groupby('date')
    retavg = grouped['minute_return'].mean()
    result_df = retavg.to_frame().reset_index()
    result_df.columns = ['date', 'retavg']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['retavg'].fillna(result_df['retavg'].mean(), inplace=True)
    return result_df


def ret_std(df: pd.DataFrame):
    grouped = df.groupby('date')
    retstd = grouped['minute_return'].std()
    result_df = retstd.to_frame().reset_index()
    result_df.columns = ['date', 'retstd']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['retstd'].fillna(result_df['retstd'].mean(), inplace=True)
    return result_df


def ret_skew(df: pd.DataFrame):
    grouped = df.groupby('date')
    retskew = grouped['minute_return'].skew()
    result_df = retskew.to_frame().reset_index()
    result_df.columns = ['date', 'retskew']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['retskew'].fillna(result_df['retskew'].mean(), inplace=True)
    return result_df


def ret_kurt(df: pd.DataFrame):
    grouped = df.groupby('date')
    kurtosis_list = []
    for date, group in grouped:
        minute_return = group['minute_return']
        if not minute_return.empty:
            kurt = kurtosis(minute_return)
        else:
            kurt = None
        kurtosis_list.append({'date': date, 'retkurt': kurt})
    result_df = pd.DataFrame(kurtosis_list)
    # 计算峰度的均值，不包括NaN值
    mean_kurtosis = result_df['retkurt'].mean(skipna=True)
    # 将峰度为NaN的值替换为均值
    result_df['retkurt'].fillna(mean_kurtosis, inplace=True)
    return result_df


# 换手率类
def swap_avg(df: pd.DataFrame):
    grouped = df.groupby('date')
    swapavg = grouped['minute_swap_rate'].mean()
    result_df = swapavg.to_frame().reset_index()
    result_df.columns = ['date', 'swapavg']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['swapavg'].fillna(result_df['swapavg'].mean(), inplace=True)
    return result_df


def swap_std(df: pd.DataFrame):
    grouped = df.groupby('date')
    swapstd = grouped['minute_swap_rate'].std()
    result_df = swapstd.to_frame().reset_index()
    result_df.columns = ['date', 'swapstd']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['swapstd'].fillna(result_df['swapstd'].mean(), inplace=True)
    return result_df


def swap_skew(df: pd.DataFrame):
    grouped = df.groupby('date')
    swapskew = grouped['minute_swap_rate'].skew()
    result_df = swapskew.to_frame().reset_index()
    result_df.columns = ['date', 'swapskew']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['swapskew'].fillna(result_df['swapskew'].mean(), inplace=True)
    return result_df


def swap_kurt(df: pd.DataFrame):
    grouped = df.groupby('date')
    kurtosis_list = []
    for date, group in grouped:
        minute_swap_rate = group['minute_swap_rate']
        if not minute_swap_rate.empty:
            kurt = kurtosis(minute_swap_rate)
        else:
            kurt = None
        kurtosis_list.append({'date': date, 'swapkurt': kurt})
    result_df = pd.DataFrame(kurtosis_list)
    # 计算峰度的均值，不包括NaN值
    mean_kurtosis = result_df['swapkurt'].mean(skipna=True)
    # 将峰度为NaN的值替换为均值
    result_df['swapkurt'].fillna(mean_kurtosis, inplace=True)
    return result_df


# 价格类
def price_avg(df: pd.DataFrame):
    grouped = df.groupby('date')
    priceavg = grouped['stand_close'].mean()
    result_df = priceavg.to_frame().reset_index()
    result_df.columns = ['date', 'priceavg']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['priceavg'].fillna(result_df['priceavg'].mean(), inplace=True)
    return result_df


def price_std(df: pd.DataFrame):
    grouped = df.groupby('date')
    pricestd = grouped['stand_close'].std()
    result_df = pricestd.to_frame().reset_index()
    result_df.columns = ['date', 'pricestd']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['pricestd'].fillna(result_df['pricestd'].mean(), inplace=True)
    return result_df


def price_skew(df: pd.DataFrame):
    grouped = df.groupby('date')
    priceskew = grouped['stand_close'].skew()
    result_df = priceskew.to_frame().reset_index()
    result_df.columns = ['date', 'priceskew']
    result_df.reset_index(drop=True, inplace=True)
    # 使用fillna方法将缺失值填充为平均值
    result_df['priceskew'].fillna(result_df['priceskew'].mean(), inplace=True)
    return result_df


def price_kurt(df: pd.DataFrame):
    grouped = df.groupby('date')
    kurtosis_list = []
    for date, group in grouped:
        stand_close = group['stand_close']
        if not stand_close.empty:
            kurt = kurtosis(stand_close)
        else:
            kurt = None
        kurtosis_list.append({'date': date, 'pricekurt': kurt})
    result_df = pd.DataFrame(kurtosis_list)
    # 计算峰度的均值，不包括NaN值
    mean_kurtosis = result_df['pricekurt'].mean(skipna=True)
    # 将峰度为NaN的值替换为均值
    result_df['pricekurt'].fillna(mean_kurtosis, inplace=True)
    return result_df


# 收盘价与换手率的相关系数
def corr_priceswap(df: pd.DataFrame):
    np.seterr(divide='ignore', invalid='ignore')
    grouped = df.groupby('date')
    # 分别计算每个分组的相关系数
    correlations = [group['close'].corr(group['minute_swap_rate']) for name, group in grouped]
    # 计算相关系数的均值
    mean_correlation = pd.Series(correlations).mean()
    # 将缺失的相关系数值替换为均值
    filled_correlations = [corr if not pd.isna(corr) else mean_correlation for corr in correlations]
    # 创建一个新的DataFrame来存储结果
    result_df = pd.DataFrame({'date': grouped.groups.keys(), 'corrpriceswap': filled_correlations})
    np.seterr(divide='warn', invalid='warn')
    return result_df


# 收益率与换手率一阶差分的相关系数
def corr_deltaswap(df: pd.DataFrame):
    np.seterr(divide='ignore', invalid='ignore')
    grouped = df.groupby('date')
    dfff_swap = grouped['minute_swap_rate'].diff()
    # 分别计算每个分组的相关系数
    correlations = [group['minute_return'].corr(dfff_swap) for name, group in grouped]
    # 计算相关系数的均值
    mean_correlation = pd.Series(correlations).mean()
    # 将缺失的相关系数值替换为均值
    filled_correlations = [corr if not pd.isna(corr) else mean_correlation for corr in correlations]
    # 创建一个新的DataFrame来存储结果
    result_df = pd.DataFrame({'date': grouped.groups.keys(), 'corrdeltaswap': filled_correlations})
    np.seterr(divide='warn', invalid='warn')
    return result_df


# 筛选并删除个股数据中，日成交总额小于500万元对应的数据
def liquid_select(df: pd.DataFrame):
    grouped = df.groupby('date')['turnover'].sum().reset_index()
    # 找到成交额小于5000000的交易日
    low_turnover_dates = grouped[grouped['turnover'] < 5000000]['date']
    # 使用isin方法创建布尔索引删除这些交易日的数据
    result_df = df[~df['date'].isin(low_turnover_dates)]
    return result_df


# 以下为先按时间将全天分时数据分为开盘（9：31-10：00）、盘中（10：01-14：30）、尾盘（14：31-15：00），在计算以上的日频因子值
# 时间范围判断函数
def time_judge(time):
    if pd.to_datetime('09:31', format='%H:%M').time() <= time <= pd.to_datetime('10:00', format='%H:%M').time():
        return 'open'
    elif pd.to_datetime('10:01', format='%H:%M').time() <= time <= pd.to_datetime('14:30', format='%H:%M').time():
        return 'intra'
    elif pd.to_datetime('14:31', format='%H:%M').time() <= time <= pd.to_datetime('15:00', format='%H:%M').time():
        return 'close'


# 创建一个函数来添加时间范围属性
def get_time_range(data: pd.DataFrame):
    # 将时间列转换为时间类型
    data['time'] = pd.to_datetime(data['time'], format='%H:%M').dt.time
    data['time_range'] = data['time'].apply(time_judge)
    return data


def get_price_range(data: pd.DataFrame):
    # 创建价格范围属性的设置与判断的函数
    def categorize_price(group):
        sorted_group = group.sort_values(by='close')
        num_rows = len(sorted_group)
        median_index = num_rows // 2
        group['price_range'] = ['high' if idx >= median_index else 'low' for idx in range(num_rows)]
        return group
    # 分组并应用划分价格区域的函数
    result_df = data.groupby('date', group_keys=False).apply(categorize_price).reset_index(drop=True)
    return result_df


# 以9：31的开盘价为基准，计算单个股票一年中的每个交易日的收益率序列
def daily_return(year: int, symbol: str):
    data = read_data(year, symbol)
    grouped = data.groupby('date')
    first_open_values = grouped['open'].first().reset_index()
    first_open_values['daily_return'] = first_open_values['open'] / first_open_values['open'].shift(1) - 1
    first_open_values.loc[first_open_values.index[0], 'daily_return'] = 0
    first_open_values.drop('open', axis=1, inplace=True)
    return first_open_values


# 三倍标准差去极值
def three_sigma_outlier(df: pd.DataFrame, column_name: str):
    # 定义上下限，三倍标准差范围
    mean = df[column_name].mean()
    std = df[column_name].std()
    lower_limit = mean - 3 * std
    upper_limit = mean + 3 * std
    # 使用布尔索引去除超出上下限的异常值
    result_df = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]
    return result_df


# 0-1标准化
def min_max_scaling(df: pd.DataFrame, column_name: str):
    min = df[column_name].min()
    max = df[column_name].max()
    df[column_name] = (df[column_name] - min) / (max - min)
    return df


