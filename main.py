from time import time
import data_processing as dp
from analysis import run_analysis
# from models.misc import Mse
import visualisation as vis
import pandas as pd
import numpy as np

def main():

    # Load Data
    # rv = dp.load_rv('data/SP500_RQ_5min.csv')
    rv_all = dp.load_rv_all('data/rv_dataset.csv')
    rv = dp.load_rv_one('data/rv_dataset.csv', '.SPX')
    # # rv_ma5 = dp.ma_rv(rv, 5)
    # # rv_ma22 = dp.ma_rv(rv, 12)
    rv_dict = {'SPX': rv}

    # Plot all RVs
    # vis.plot_all_rv(rv_dict)
    # vis.plot_series(rv)

    # Run analysis for all rv series
    for i, rv in rv_all.items():
        print(f'Analysis for {i}')

        # Run Analysis(Forecast) and Generate Predictions
        df = run_analysis(rv)

        # # Results
        # pred_cols = ['rough_har_pred', 'har_pred', 'roughvol_fc_d']
        # # pred_cols = ['rough_har_pred', 'har_pred', 'c_pred', 'roughvol_fc_d']

        # # # RMSE
        # for p in pred_cols:
        #     Mse(df.iloc[900:], 'y', p).print_rmse()

        # # Forecast Plots
        # for p in pred_cols:
        #     vis.plot_forecast(df, p)
        #     # vis.plot_forecast_returns(df, p)
        
        # vis.plot_series(np.array(df['roughvol_fc_d']) - np.array(df['har_pred']))

    return 0

if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")