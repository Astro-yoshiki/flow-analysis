import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adam

# setting for beautiful graph
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.3


class NN:
    def __init__(self, data_path=None, validation_path=None, save_path=None):
        self.data_path = data_path
        self.validation_path = validation_path
        if save_path is None:
            save_path = "result/"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        self.save_path = save_path

        self.x = None
        self.x_train_std = None
        self.x_test_std = None
        self.y = None
        self.y_train_std = None
        self.y_test = None
        self.y_test_std = None
        self.y_test_pred = None
        self.xlabel = None
        self.ylabel = None
        self.xdim = None
        self.ydim = None
        self.x_scaler = None
        self.y_scaler = None
        self.model = None
        self.history = None
        
    def load_data(self, input_output_boundary=None):
        df = pd.read_csv(self.data_path)
        self.x = df.iloc[:, :input_output_boundary]
        self.xdim = self.x.shape[1]
        self.xlabel = self.x.columns
        self.y = df.iloc[:, input_output_boundary:]
        self.ydim = self.y.shape[1]
        self.ylabel = self.y.columns

        # FIXME: 構造データだからランダムに分けない！
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=None)
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        self.x_train_std = x_scaler.fit_transform(x_train)
        self.x_test_std = x_scaler.transform(x_test)
        self.y_train_std = y_scaler.fit_transform(y_train)
        self.y_test_std = y_scaler.transform(y_test)

        # スケーラーの保存
        dump(x_scaler, self.save_path + "x_scaler.joblib")
        dump(y_scaler, self.save_path + "y_scaler.joblib")
        
        # スケーラーをクラス変数に属させる
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        
    def build_model(self, num_node=128, num_hidden_layer=1):
        self.model = Sequential()
        self.model.add(Dense(num_node, input_dim=self.xdim, activation="relu"))
        for layer in range(num_hidden_layer):
            self.model.add(Dense(num_node, activation="relu"))
        self.model.add(Dense(self.ydim, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=1e-4), metrics=["mae"])
        self.model.summary()
    
    def fit(self, epochs=1000, batch_size=256):
        self.history = self.model.fit(self.x_train_std, self.y_train_std, epochs=epochs, batch_size=batch_size,
                                      validation_split=0.1, verbose=1)
        # モデルの保存
        open(self.save_path + "model.json", "w").write(self.model.to_json())
        # 学習済みの重みを保存
        self.model.save_weights(self.save_path + "weights.h5")
        
    def plot_history(self):
        mae = self.history.history["mae"]
        val_mae = self.history.history["val_mae"]
        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]
        epochs = range(1, len(mae)+1)

        fig = plt.figure(figsize=(10, 5))
        # MAEのプロット
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(epochs, mae, "b", label="Training mae")
        ax1.plot(epochs, val_mae, "r", label="Validation mae")
        ax1.set_yscale("log")
        ax1.grid()
        ax1.set_title("MAE")
        ax1.legend(loc="best")

        # Lossのプロット
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(epochs, loss, "b", label="Training loss")
        ax2.plot(epochs, val_loss, "r", label="Validation loss")
        ax2.set_yscale("log")
        ax2.grid()
        ax2.set_title("Loss")
        ax2.legend(loc="best")

        plt.savefig(self.save_path + "mae_loss.png", bbox_inches="tight")
        plt.close()
        
    def plot_interrelation(self, use_saved_model=False, model_path=None, weights_path=None):
        # evaluation of test
        if use_saved_model and model_path is not None and weights_path is not None:
            # モデルの読み込み
            self.model = open(model_path).read()
            self.model = model_from_json(self.model)
            self.model.load_weights(weights_path)
        else:
            pass
        y_test_pred_std = self.model.predict(self.x_test_std)
        self.y_test_pred = self.y_scaler.inverse_transform(y_test_pred_std)
        self.y_test = self.y_scaler.inverse_transform(self.y_test_std)
        
        plt.figure(figsize=(10, 5))
        plt.subplots_adjust(wspace=0.6)
        for i in range(len(self.ylabel)):
            plt.subplot(1, len(self.ylabel), i+1)
            plt.scatter(self.y_test[:, i], self.y_test_pred[:, i], s=5)
            plt.plot([min(self.y_test[:, i]), max(self.y_test[:, i])], [min(self.y_test[:, i]), max(self.y_test[:, i])],
                     color="black", linestyle="dashed")
            plt.title("Parity-plot of " + self.ylabel[i])
            plt.xlabel("Simulation")
            plt.ylabel("Machine Learning")
            
            # Rスコアを出力
            r2 = r2_score(self.y_test[:, i], self.y_test_pred[:, i])
            print("R2 score of {0} : {1:.5f}".format(self.ylabel[i], r2))

        plt.savefig(self.save_path + "parity-plot.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    reuse_model = True
    _data_path = "../Data_for_ML/Growth_cell.csv"
    _validation_path = "../Data_for_ML/Validation.csv"
    _input_output_boundary = 7
    if reuse_model:
        nn = NN(data_path=_data_path, validation_path=_validation_path)
        nn.load_data(input_output_boundary=_input_output_boundary)
        nn.plot_interrelation(use_saved_model=True, model_path="result/model.json", weights_path="result/weights.h5")
    else:
        nn = NN(data_path=_data_path, validation_path=_validation_path)
        nn.load_data(input_output_boundary=_input_output_boundary)
        nn.build_model(num_node=128, num_hidden_layer=1)
        nn.fit(epochs=2000, batch_size=256)
        nn.plot_history()
        nn.plot_interrelation()
