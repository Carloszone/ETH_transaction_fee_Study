from cryptoapp.data import list_file, read_file
from datetime import date
import pandas as pd

def weekend_check(_datetime):
  if date.weekday(_datetime) >=5:
    res = 1
  else:
    res = 0
  return res

def merge_df(data_type = 'price'):
    files_list = list_file(data_type)
    df_list = []
    for file in files_list:
        df_list.append(read_file(file))
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

def filter_df(merged_df, var_list):
    ind = merged_df.Name.isin(var_list)
    return merged_df[ind]

def pivot_df(df):
    return df.pivot(index='Date', columns='Name', values=['Open', 'High', 'Low', 'Close'])

def validation_df(input, summary = False):
    df = input.copy()
    # na check
    missing = df.isna().sum().sort_values(ascending=False)
    percent_missing = ((missing / df.isnull().count()) * 100).sort_values(ascending=False)
    missing_df = pd.concat([missing, percent_missing], axis=1, keys=['Total', 'Percent'], sort=False)

    # fill na
    columns = list(missing_df[missing_df['Total'] >= 1].reset_index()['index'])

    for col in columns:
        null_index = df.index[df[col].isnull() == True].tolist()
        null_index.sort()
        for ind in null_index:
            if ind > 0:
                df.loc[ind, col] = df.loc[ind - 1, col]
            if ind == 0:
                df.loc[ind, col] = 0

    # outliers check
    count = []
    for col in df.columns:
        count.append(sum(df[col] > df[col].mean() + 2 * df[col].std()) + sum(df[col] < df[col].mean() - 2 * df[col].std()))
    outliers_df = pd.DataFrame({'Columns': df.columns, 'Count': count}).sort_values(by = 'Count')

    if summary == True:
        print('missing value check:/n')
        print(missing_df)
        print('/n outliers check:/n')
        print(outliers_df)

    return df

def join_df(var_list):

    transaction_df = merge_df(data_type='transaction')

    price_df = merge_df(data_type = 'price')
    price_df = filter_df(price_df, var_list)
    price_df = pivot_df(price_df)

    price_df.columns = price_df.columns.map('.'.join)
    price_df = price_df.reset_index()

    transaction_df['Date'] = pd.to_datetime(transaction_df['Date'], format = '%Y-%m-%d')
    price_df['Date'] = pd.to_datetime(price_df['Date'], format = '%Y-%m-%d')



    joined_df = transaction_df.merge(price_df, how = 'left', on = 'Date')
    joined_df = validation_df(joined_df)
    joined_df = joined_df.set_index('Date')
    return joined_df
