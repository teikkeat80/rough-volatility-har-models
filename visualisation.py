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

def plot_comparison_2(first, second, first_label, second_label):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(first)), first, 'b-', label=first_label)
    plt.plot(np.arange(len(second)), second, 'r-', label=second_label)
    plt.xlabel('Time')
    plt.legend()
    plt.show()

def plot_comparison_3(first, second, third, first_label, second_label, third_label):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(first)), first, 'b-', label=first_label)
    plt.plot(np.arange(len(second)), second, 'r-', label=second_label)
    plt.plot(np.arange(len(third)), third, 'k-', label=third_label)
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

def plot_acorr(array):
    plt.figure(figsize=(10, 6))
    plt.acorr(array, usevlines=False, maxlags=100, linestyle='-', marker=' ')
    plt.xlim([0, 100])
    plt.ylim([0, 1])
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