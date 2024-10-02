from data_processing import load_rv
from analysis import run_analysis
from models.misc import Mse
from visualisation import plot_forecast
from time import time

def main():

    # Load Data
    rv = load_rv('data/SNP500_RV_5min.csv')

    # Run Analysis(Forecast) and Generate Predictions
    df = run_analysis(rv)

    # Results
    pred_cols = ['rough_har_pred', 'har_pred', 'comb_pred', 'roughvol_fc_d']

    # RMSE
    for p in pred_cols:
        Mse(df, 'y', p).print_rmse()

    # Forecast Plots
    for p in pred_cols:
        plot_forecast(df, p)

    return 0

if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f"Elapsed time: {end_time - start_time} seconds")