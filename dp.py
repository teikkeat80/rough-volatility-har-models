import pandas as pd

def load_spxfuturesrv():
    df_raw = pd.read_csv('data/spxfuturesrv.csv')
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted['RV'].tolist()
    return rv

def load_spxfuturesrq():
    df_raw = pd.read_csv('data/spxfuturesrq.csv')
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rq = df_sorted['RQ'].tolist()
    return rq

def load_marketspotrv():
    df_raw = pd.read_csv('data/marketspotrv.csv')
    rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
    return rv_all

def ma(series, window):
    pd_series = pd.Series(series)
    ma_series = pd_series.rolling(window).mean()
    return ma_series.dropna().tolist()