from matplotlib import pyplot as plt
from hurst import Hurst
import numpy as np

def plot_scaling_diagram(h_instance: Hurst):
    h_instance.plot_scale_m_delta()
    h_instance.plot_scale_zeta_q()

def plot_forecast(df, model):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['y'], label='RV')
    plt.plot(df.index, df[model], label=model)
    plt.title(f'Forecast Plot of RV - {model}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison(actual, predicted, model):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(actual)), actual, label='actual')
    plt.plot(np.arange(len(predicted)), predicted, label=model)
    plt.title(f'Forecast Plot of RV - {model}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_forecast_returns(df, model):
    returns = np.array(df[model]) * np.random.normal(0, 1, len(df[model]))
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, returns, label=model)
    plt.title(f'Forecast Plot of Returns Based on {model}')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d(x, y, z):
    plt.figure().add_subplot(projection='3d').scatter(x, y, z)
    plt.show()

def plot_all_rv(dict):
    for key, val in dict.items():
        plt.figure(figsize=(10, 6))
        plt.plot(val, label=key)
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

def plot_series(array):
    plt.figure(figsize=(10, 6))
    plt.plot(array)
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()