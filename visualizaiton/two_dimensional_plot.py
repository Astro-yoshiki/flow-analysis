import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from tensorflow.keras.models import model_from_json

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


class Plotter:
    def __init__(self, ref_path=None):
        if ref_path is None:
            ref_path = "../modeling/result/"
        self.ref_path = ref_path

        self.x = None
        self.y = None
        self.xx = None
        self.yy = None
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.x_resolution = None
        self.y_resolution = None

        self.t = None
        self.u = None
        self.v = None

    def load_model_and_scaler(self):
        # モデルの読み込み
        model_path = self.ref_path + "model.json"
        weights_path = self.ref_path + "weights.h5"
        self.model = open(model_path).read()
        self.model = model_from_json(self.model)
        self.model.load_weights(weights_path)

        # スケーラの読み込み
        self.x_scaler = load(self.ref_path + "x_scaler.joblib")
        self.y_scaler = load(self.ref_path + "y_scaler.joblib")

    def make_grid(self, xl, xu, x_resolution, yl, yu, y_resolution, return_grid=False):
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.x = np.linspace(xl, xu, self.x_resolution)
        self.y = np.linspace(yl, yu, self.y_resolution)
        # メッシュグリッドを作成
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        # 必要なら返す
        if return_grid:
            return self.xx, self.yy

    def predict_point(self, x):
        x = np.array(x).reshape(1, -1)
        x_std = self.x_scaler.transform(x)
        y_pred_std = self.model.predict(x_std)
        y_pred = self.y_scaler.inverse_transform(y_pred_std)
        return y_pred

    def predict_array(self, x):
        x = np.array(x)
        x_std = self.x_scaler.transform(x)
        y_pred_std = self.model.predict(x_std)
        y_pred = self.y_scaler.inverse_transform(y_pred_std)
        return y_pred

    def predict_one_cell(self, params: list):
        self.t = np.zeros((len(self.xx), len(self.yy)))
        self.u = np.zeros((len(self.xx), len(self.yy)))
        self.v = np.zeros((len(self.xx), len(self.yy)))

        # 各グリッドの予測
        for i in range(len(self.xx)):
            for j in range(len(self.yy)):
                cond = params + [self.xx[i][j]] + [self.yy[i][j]]
                self.t[i][j] = self.predict_point(cond)[:, 0] + 273.15
                self.u[i][j] = self.predict_point(cond)[:, 1]
                self.v[i][j] = self.predict_point(cond)[:, 2]

    def predict_all_cell(self, params: list, return_prediction=False):
        # 全てのグリッドを用意しておく
        grid_x = np.tile(self.x, self.x_resolution).reshape(-1, 1)
        grid_y = []
        for i in range(len(self.y)):
            for j in range(self.y_resolution):
                grid_y.append(self.y[i])
        grid_y = np.array(grid_y).reshape(-1, 1)
        # グリッド分だけパラメータを複製
        params_grid = np.array(params)
        params_grid = np.full((grid_x.shape[0], len(params)), params_grid)
        # 条件と座標を合体させる
        all_params = np.hstack([params_grid, grid_x, grid_y])

        # モデルによりまとめて予測
        self.t = self.predict_array(all_params)[:, 0] + 273.15
        self.u = self.predict_array(all_params)[:, 1]
        self.v = self.predict_array(all_params)[:, 2]
        # 座標と対応づけられる形にreshape
        self.t = self.t.reshape(self.x_resolution, self.y_resolution)
        self.u = self.u.reshape(self.x_resolution, self.y_resolution)
        self.v = self.v.reshape(self.x_resolution, self.y_resolution)

        # 必要なら返す
        if return_prediction:
            return self.t, self.u, self.v

    def plot_2d(self, figsize=(20, 10)):
        cmap = plt.get_cmap("jet")
        fig, (ax0, ax1, ax2) = plt.subplots(figsize=figsize, nrows=1, ncols=3)

        im0 = ax0.pcolormesh(self.xx, self.yy, self.t, cmap=cmap, shading="gouraud")
        fig.colorbar(im0, ax=ax0)
        ax0.set_xlabel("$X [mm]$")
        ax0.set_ylabel("$Y [mm]$")
        ax0.set_title("$T [K]$")

        im1 = ax1.pcolormesh(self.xx, self.yy, self.u, cmap=cmap, shading="gouraud")
        fig.colorbar(im1, ax=ax1)
        ax1.set_xlabel("$X [mm]$")
        ax1.set_ylabel("$Y [mm]$")
        ax1.set_title("$U [m/s]$")

        im2 = ax2.pcolormesh(self.xx, self.yy, self.v, cmap=cmap, shading="gouraud")
        fig.colorbar(im2, ax=ax2)
        ax2.set_xlabel("$X [mm]$")
        ax2.set_ylabel("$Y [mm]$")
        ax2.set_title("$V [m/s]$")

        fig.tight_layout()
        plt.savefig("2D-plot.png", bbox_inches="tight")
        plt.close()

    def plot_2d_streamplot(self):
        cmap = plt.get_cmap("jet")
        plt.pcolormesh(self.xx, self.yy, self.t, cmap=cmap, shading="gouraud")
        plt.xlabel("$X [mm]$")
        plt.ylabel("$Y [mm]$")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar()
        # 配列要素を3つ飛ばしで描画（::3）
        plt.streamplot(self.xx[::3, ::3], self.yy[::3, ::3], self.u[::3, ::3], self.v[::3, ::3],
                       color=np.sqrt(self.u[::3, ::3] ** 2 + self.v[::3, ::3] ** 2), cmap="binary_r")
        # ベクトル場の大きさで流線の色をつける
        # plt.colorbar()
        plt.savefig("2D-streamplot.png", bbox_inches="tight")
        plt.close()

    def plot_2d_quiverplot(self):
        cmap = plt.get_cmap("jet")
        plt.pcolormesh(self.xx, self.yy, self.t, cmap=cmap, shading="gouraud")
        plt.xlabel("$X [mm]$")
        plt.ylabel("$Y [mm]$")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar()
        # 配列要素を2つ飛ばしで描画（::2）
        plt.quiver(self.xx[::2, ::2], self.yy[::2, ::2], self.u[::2, ::2], self.v[::2, ::2],
                   cmap="binary_r", width=0.0075)
        plt.savefig("2D-quiverplot.png", bbox_inches="tight")
        plt.close()

    def plot_2d__for_animation(self, figsize=(7, 7)):
        cmap = plt.get_cmap("jet")
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(self.xx, self.yy, self.t, cmap=cmap, shading="gouraud")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label="$T [K]$", rotation=0, labelpad=20)
        ax.set_xlabel("$X [mm]$")
        ax.set_ylabel("$Y [mm]$")
        # 配列要素を2つ飛ばしで描画（::2）
        ax.quiver(self.xx[::2, ::2], self.yy[::2, ::2], self.u[::2, ::2], self.v[::2, ::2],
                  cmap="binary_r", width=0.0075)
        plt.savefig("2D-quiverplot_for_animation.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    plotter = Plotter()
    plotter.load_model_and_scaler()
    plotter.make_grid(0.0, 76.2, 30, 135.0, 239.5, 30)

    _params = [2000.701993, 1202.915643, 27.920038, 35.147359, 28.187168]
    plotter.predict_all_cell(params=_params)
    plotter.plot_2d_quiverplot()
