import pandas as pd

def load_rv(path):
    df_raw = pd.read_csv(path)
    df_sorted = df_raw.sort_values(by='Date', ignore_index=True)
    rv = df_sorted['RV'].tolist()
    return rv