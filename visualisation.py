from matplotlib import pyplot as plt
from hurst import Hurst
import numpy as np
import seaborn as sb

def plot_scaling_diagram(h_instance: Hurst):
    h_instance.plot_scale_m_delta()
    h_instance.plot_scale_zeta_q()

def plot_comparison(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(actual)), actual, 'k-', label=r'$RV_t$', linestyle=(0, (3, 1, 1, 1)), lw=0.8)
    plt.plot(np.arange(len(predicted)), predicted, 'r-', label=r'$\hat{RV_t}$')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def plot_series(array, ylab):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(array)), array, 'k-')
    plt.xlabel('Time')
    plt.ylabel(ylab)
    plt.show()

def plot_superimpose_series(array, label):
    plt.figure(figsize=(10, 6))
    for i, j in enumerate(array):
        plt.plot(np.arange(len(j)), j, label=label[i])
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def plot_acorr(array, ylim):
    plt.figure(figsize=(10, 6))
    plt.acorr(array, usevlines=False, maxlags=100, linestyle='-', marker=' ')
    plt.xlim([0, 100])
    plt.ylim([ylim, 1])
    plt.xlabel('lag')
    plt.ylabel('$ACF$')
    plt.show()

def plot_kd(array):
    plt.figure(figsize=(10, 6))
    sb.kdeplot(array, color='grey', shade=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

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