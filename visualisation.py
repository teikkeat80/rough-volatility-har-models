from models.hurst import Hurst
from matplotlib import pyplot as plt

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