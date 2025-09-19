import pandas as pd
import factor_calculators
import inspect


price_vol_data_file_path = 'data/sliced/all_stocls_daily_price_vol_sliced_300_20180101_20201231'
factor_output_path = 'data/sliced/all_stocls_daily_factors_300_20180101_20201231'

data = pd.read_parquet(price_vol_data_file_path)

all_factor_data = data[['trade_date', 'ts_code']]

factor_funcs = [
        (name, fn) 
        for name, fn in inspect.getmembers(factor_calculators, inspect.isfunction)
        if name.startswith("alpha")
    ]

for name, fn in factor_funcs:
    setting = {
        
        'factor_name': name
    }

    factor_data = fn(data.copy())
    # wide_df: index=trade_date, columns=ts_code, values=alpha21
    long_df = (
        factor_data
        .stack()                # Series with MultiIndex (trade_date, ts_code)
        .rename(setting['factor_name'])      # name the series
        .reset_index()          # turn index levels into columns
    )

    # rename columns if needed
    long_df.columns = ['trade_date','ts_code',setting['factor_name']]

    all_factor_data = pd.merge(all_factor_data, long_df, on=['trade_date', 'ts_code'], how='left')


all_factor_data.to_parquet(factor_output_path)