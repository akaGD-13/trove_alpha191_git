import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm


# %%
def alpha21(data):
    '''
    通过输入的数据集生成一个特定因子,  逻辑是用过去6日的6日均值序列(实际上使用了11日数据),
    对[1,2,3,4,5,6]这个序列做回归, 对它做回归的含义是, 以[1,2,3,4,5,6]作为自变量.
    最终因子为这个回归的回归系数.
    本因子对线性回归的计算基于sklearn, 计算时间大约需要50分钟

    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    model = LinearRegression()
    X = np.arange(1, 7).reshape(-1, 1)

    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')

    factor1 = close_panel.rolling(6).mean()

    df = pd.DataFrame(index=factor1.index, columns=factor1.columns)

    for row in tqdm(range(5, close_panel.shape[0])):
        for col in range(close_panel.shape[1]):
            # 需要至少6个数据点
            Y = close_panel.iloc[row - 5:row + 1, col].values.reshape(-1, 1)  # 目标变量
            if np.isnan(Y).sum() != 0:
                continue
            model.fit(X, Y)  # 拟合回归模型
            df.iloc[row, col] = model.coef_[0, 0]  # 取回归系数（斜率）

    return df


def alpha24(data):
    '''
    通过输入的数据集生成一个特定因子, 构建逻辑如下: 
        因子核心为close-delay(close, 5)
        我们以(5,1)的参数计算这个因子核心的SMA. 
        计算时间预计需要30分钟. 
        
    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
    
    factor = close_panel - close_panel.shift(5)

    df = pd.DataFrame(index=factor.index, columns=factor.columns)
    
    for row in tqdm(range(1, factor.shape[0])):
        for col in range(factor.shape[1]):
            Y_i = df.iloc[row-1, col]
            A = factor.iloc[row, col]
            if not np.isnan(A) and np.isnan(Y_i):
                Y = A
            elif np.isnan(A):
                Y = np.nan
            else:
                Y = A / 5 + 4 * Y_i / 5
            df.iloc[row, col] = Y
    
    return df


def alpha27(data):
    '''
    通过输入的数据集生成一个特定因子, 因子核心包括两部分: 
        1. close/delay(close, 3) - 1
        2. close / delay(close, 6) - 1
    将3日涨跌幅和6日涨跌幅相加*100, 然后计算其参数为12的WMA
    WMA的构建方式是, 对序列以0.9的指数衰减加权.
        
        
    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close

    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理

    '''
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
    factor1 = close_panel / close_panel.shift(3) - 1
    factor2 = close_panel / close_panel.shift(6) - 1
    factor = (factor1 + factor2) * 100
    
    weight = []
    for i in range(12, 0, -1):
        weight.append(0.9 ** i)
    weight = np.array(weight) / sum(weight)
    
    df = factor.rolling(window=12).apply(lambda x: np.sum(x * weight[-len(x):]), raw=True)
    
    return df


def alpha28(data):
    '''
    通过输入的数据集生成一个特定因子, 因子核心为: 
        收盘价与九日最低价的最小值之差与九日最高价的最大值和九日最低价的最小值之差的比
        对这一核心*100, 计算(3,1)的SMA
        factor1为(3,1)的SMA
        factor2为factor1参数(3,1)的SMA
        最终因子为3*factor1 - 2*factor2
        
    因子计算预计需要一个小时30分钟. 
        
        
    Parameters
    ----------
    data : DataFrame
        输入一个已经经过计算的数据集, 本因子至少需要输入close, high, low
    Returns
    -------
    输出一个各交易日收盘后可计算得到的因子值面板
    备注: 为方便输出为excel, 对输出面板的时间序列不做datetime处理
            
    '''
    close_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='close')
    high_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='high')
    low_panel = pd.pivot(data=data, columns='ts_code', index='trade_date', values='low')
    
    factor = (close_panel - low_panel.rolling(9).min()) / (high_panel.rolling(9).max() - low_panel.rolling(9).min()) * 100
    
    factor1 = pd.DataFrame(index=factor.index, columns=factor.columns)
    
    for row in tqdm(range(1, factor.shape[0])):
        for col in range(factor.shape[1]):
            Y_i = factor1.iloc[row-1, col]
            A = factor.iloc[row, col]
            if not np.isnan(A) and np.isnan(Y_i):
                Y = A
            elif np.isnan(A):
                Y = np.nan
            else:
                Y = A / 3 + 2 * Y_i / 3
            factor1.iloc[row, col] = Y
            
    factor2 = pd.DataFrame(index=factor.index, columns=factor.columns)  
      
    for row in tqdm(range(1, factor1.shape[0])):
        for col in range(factor1.shape[1]):
            Y_i = factor2.iloc[row-1, col]
            A = factor1.iloc[row, col]
            if not np.isnan(A) and np.isnan(Y_i):
                Y = A
            elif np.isnan(A):
                Y = np.nan
            else:
                Y = A / 3 + 2 * Y_i / 3
            factor2.iloc[row, col] = Y
            
    df = 3 * factor1 - 2 * factor2
    
    return df