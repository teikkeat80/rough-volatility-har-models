import numpy as np

# MSE Calculation
class Mse:
    def __init__(self, df, act: str, pred: str):
        self.model = pred
        self.mse = np.mean((df[act] - df[pred]) ** 2)
        self.rmse = round(np.sqrt(self.mse), 6)

    def print_rmse(self):
        output = self.model + " model: " + str(self.rmse)
        print(output)
        return(output)