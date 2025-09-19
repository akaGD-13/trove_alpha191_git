import pandas as pd


def alpha0(data: pd.DataFrame) -> pd.DataFrame:
    # work on a sorted copy
    df = data.sort_values(['ts_code','trade_date']).copy()
    df['pre_close'] = df.groupby('ts_code')['close'].shift(1)
    df['alpha0'] = df['close'] / df['pre_close']
    
    return df[['trade_date','ts_code','alpha0']]

def alpha000(data: pd.DataFrame) -> pd.DataFrame:
    df = data.sort_values(['ts_code','trade_date']).copy()
    df['alpha000'] = df['pe'] / df['vol']

    return df[['trade_date','ts_code','alpha000']]