import pandas as pd


start_date = 20180101
end_date = 20201231
ts_code_range = 300
input_file_folder = 'data/'
input_file_name = 'all_stocls_daily_price_vol'
output_directory = 'data/sliced/'

# data = pd.read_csv('data/all_stocks_daily_with_market_cap.csv')

# data.to_parquet('data/all_stocls_daily_price_vol')

data = pd.read_parquet(input_file_folder + input_file_name)


data = data[(data['trade_date'] > start_date) & (data['trade_date'] < end_date)]
# data.dropna(axis=1, how='all', inplace=True)

codes = data['ts_code'].unique()
print('stock_codes', codes)
print('len of stock codes list', len(codes))
data = data[data['ts_code'].isin(codes[:ts_code_range])]

print('number of null values', data['close'].isna())

output_file_path = output_directory + input_file_name + '_sliced_' + str(ts_code_range) + '_' + str(start_date) + '_' + str(end_date)
data.to_parquet(output_file_path)

print('sliced data saved at:', output_file_path)

neutralize_data = data[['trade_date', 'ts_code', 'total_mv']]
ind_data = pd.read_csv('data/industry_data.csv')

neutralize_data = pd.merge(neutralize_data, ind_data, on=['ts_code'], how="left")

output_file_path = output_directory + 'neutralize_data' + '_sliced_' + str(ts_code_range) + '_' + str(start_date) + '_' + str(end_date)
neutralize_data.to_parquet(output_file_path)
