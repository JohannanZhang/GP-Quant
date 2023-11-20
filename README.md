# GP-Quant
本项目采用改进后的遗传规划gplearn方法，以中证500股票标的为研究对象，对提取分时信息后的日频量价数据进行因子挖掘。
挖掘过程中以因子对应的十分组选股组合超额收益信息比率IR作为适应度，通过一定代数的变异、筛选，再经测试集的有效性检验、相关性剔除，
最终得到了8个较为有效的因子，并保存它们的结构、统计指标，以及在测试集中对应的股票组合收益图像。

一、该项目相对于原gplearn包改进的地方：
1、原gplearn包中默认算子为部分截面算子，且不可定义时序算子。本项目的custom_function.py基于原包定义了_Function类，可以定义含滚动窗口参数的时序算子。
2、原gplearn包中的公式树生成主要应用于二维数组X对一维数组Y的回归，并以R*2、残差平方和等指标作为适应度进行选择。难以应用于时序算子参与运算的更高维度变量X数据，
且判断指标有限。本项目的custom_program.py基于原包定义了由三维变量X数据至二维因子数据的计算，且定义以十分组因子选股组合IR为适应度，更符合因子筛选逻辑。

二、项目全流程：

（一）数据读取与日频变量数据计算：
读取中证500指数特定年份下所有股票的分时量价相关数据，并对股价数据进行前复权处理。这些指标用于刻画日内分钟级的市场结构。
对以上得到的股票日内分时指标数据进行统计特征提取，即对分时数据进行降频提取，最终提取出对应的84个变量，具体步骤为：

数据预处理
剔除股票日成交额低于500万元的对应分时数据
对股价进行0-1最小最大标准化，避免一字板情况使计算结果报错。
将股票分时数据，按照开盘（9：31-10：00）、盘中（10：01-14：30）、尾盘（14：31-15：00），分割为3个dataframe数据：df_open, df_intra, df_close
按照股票分时股价是否大于等于当日股价的中位数，将股票分时数据分割为2个dataframe数据：df_high, df_low。
加上股票未分割的分时数据，从而得到了每只股票的6个全年分时量价数据dataframe

日频变量值计算
对以上单只股票的6个全年分时量价数据，分别进行日内数据降频化：即求取每日分时收益率、分时换手率、标准化分时价格的的均值、标准差、偏度、峰度，
以及分时股价和换手率的相关系数（按研报要求取未标准化股价）、分时收益率和换手率一阶差分的相关系数，共计14个日频的统计指标。最终得到单只股票全年84个日频变量值。
由于目标为提取中证500指数所有标的在某一年份下的84个变量，所以采用多进程的方法，计算每一只股票在当年的84个日频变量值，并在相应文件夹下保存。从而得到500只股票全年的84个日频变量数据。

统一股票变量值矩阵的维度
受500只股票中部分股票受停牌暂停交易等因素影响，得到的500只股票对应的变量数据记录长短不一，即它们的列数为84，但由各自有效交易日数据计算而来的行数却不尽相同。难以组成一个统一的三维数组。
为了统一数据维度，便于后面计算，通过剔除掉当年有效交易交易日数不多于最多股票有效交易日90%对应的股票，并将剩余股票变量数据进行缺失行均值填充的方式，
得到了筛选后样本股票全年的84个日频变量数据array_x，它的三个维度分别为：当年交易日数、提取的股票数、变量数，以2017年数据为例，数组的维度为（244， 432， 84），
以及筛选后样本股票全年基于开盘价的日收益率二维因变量数组array_y，它的两个维度分别为：当年交易日数、提取的股票数。以2017年为例，数组的维度为（244， 432）。

（二）算子定义与因子计算
截面双变量算子（6个）：加、减、乘、除、取最大、取最小
截面单变量算子（5个）：平方、立方、绝对值开平方、开立方、绝对值求自然对数
时序算子（9个）：过去d个数据的最大值max、最小值min、平均值mean、标准差std、中位数median、x[i]-mean、(x[i] - min)/(max - min)、(x[i]-mean)/std、mean/std
通过随机生成的公式树，将以上算子，同步骤（一）得到的变量值结合，计算出样本股票全年对应的因子值（二维数据）

（三）遗传规划迭代计算与结果保存
本项目采用锦标赛选择方式，设置种群数量为200，每次比赛参与个体数量为20，最大代数为100，结合以上定义的算子与变量，基于计算得来的因子值进行十分位选股，以股票组合的超额收益IR为适应度。
迭代100代后，将每一代的个体数据进行保存，并基于每一代的最优个体，通过在测试集2015年变量数据的检验、相关性检验，得到最终最优的8个因子，将这8个因子的结构、指标与对应股票组合在测试集的表现图表进行保存。


三、项目文件介绍：

数据结果文件：
function_save文件夹：1-100代公式树群的信息（0-99_FuncTreeList），以及经过筛选后最终8个公式树的列表信息（super_ FuncTreeList），格式均为.pkl，
viable文件夹：作为训练集的2017年中证500标的股票的变量值，以及作为测试集的2015年中证500标的股票的变量值

py文件：
1.preprocess：定义读取处理本地存储的特定年份特定股票的分时数据、股价标准化、按时间、价格分割数据，以及14个计算每日变量的函数。

2.calculateVirableData：基于preprocess文件，定义将特定年份特定股票分时数据处理为含有84个变量的日频数据的函数。以及通过多进程运算，将训练集2017年、测试集2015年原始数据处理为各个股票于当年84列变量的日频二维数据，并保存本地。

3.calculateVirableData：读取基于calculateVirableData存储的，特定年份下的所有股票84列的二维数据，通过剔除缺失行过多个体、剩余个体缺失行填充的方式，得到维度为（有效交易日数，有效交易股票数，变量数）的三维变量值数组X，
以及维度为（有效交易日数，有效交易股票数）的二维数组Y，表示每只股票基于开盘价的每日收益率。X将被用于后续同公式树结合，计算因子值，Y将用于因子的筛选。

4.custom_function：基于gplearn中function函数进行了改进，定义了公式树中单个算子的类_Function。

5.GPLEARN：定义截面与时序函数，并创建了对应的算子类对象。

6.custom_program：基于gplearn中program函数进行了改进，用于定义公式树的类_Program，以及公式树的随机生成、适应度计算、变异等功能。文件中用到的函数为：
build_program 公式树列表生成。用于生成一个类对象的program属性，为一个公式树列表，由算子Function对象与常数构成。
make_ts_function 随机设置时序算子的滚动窗口参数d。
execute 公式树对应因子计算。基于类对象属性program，将公式树列表与变量数据X结合，计算该公式树对应因子的数组，数据结构为二维数据（有效交易日数，有效交易股票数）。
raw_fitness 因子选股组合的超额收益信息比率。
excess_ret 因子选股组合的全年每日超额收益率总和。
subtree_mutation 子树变异。从原先公式树中随机选择出一个子树，用一个随机生成的新子树进行替代。
hoist_mutation 简化变异。从原先公式树中随机选择出一个子树，并删除。
point_mutation 点变异。随机选择公式树的一个节点进行替换。

7.main：基于锦标赛选择法的遗传规划，计算每一代的公式树群并以.pkl文件形式保存，并筛选出每代最优个体中通过测试集检测的个体。

8.factor_select:读取本地保存的每一代公式树群，筛选每一代中适应度最高个体，并剔除与其他因子相关性大于0.7的个体，将最终有效的公式树因子保存在本地。

9.factor_display：读取最终有效的公式树因子文件，打印出每一个因子信息（公式树结构、IC、IR，以及于训练集17年，测试集15年的选股基于中证500的超额收益），并作图表现出各个因子在测试集年份中选股组合的累计收益、基准指数收益，以及超额收益。
