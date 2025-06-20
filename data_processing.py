import pandas as pd
import sys

# Bollerslev et al. (2016) data: start from 1999 consists of 4000+ records
# Oxford Man Institute data: start from October 2006 to June 2022 consists of 2000+ records

def load_rv(path, x):
    df_raw = pd.read_csv(path)
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted[x].tolist()
    return rv

def load_rv_all(path):
    df_raw = pd.read_csv(path)
    rv_all = df_raw.iloc[:, 1:].to_dict(orient='list')
    return rv_all

def load_rv_one(path, select):
    rv_all = load_rv_all(path)
    rv_select = rv_all[select]
    return rv_select

def ma_rv(rv, window):
    rv_series = pd.Series(rv)
    rv_ma = rv_series.rolling(window).mean()
    return rv_ma.dropna().tolist()