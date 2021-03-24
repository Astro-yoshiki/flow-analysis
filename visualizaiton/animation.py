import tkinter as tk
import tkinter.ttk as ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from two_dimensional_plot import Plotter

root = tk.Tk()  # ウインドの作成
root.title("PVT SiC")  # ウインドのタイトル
root.geometry("1200x750")  # ウインドの大きさ
frame_1 = tk.LabelFrame(root, labelanchor="nw", text="グラフ", foreground="green")
frame_1.grid(row=0, column=0)
frame_2 = tk.LabelFrame(root, labelanchor="nw", text="パラメータ", foreground="green")
frame_2.grid(row=0, column=1, sticky="nwse")


# スケールバーが動いたらその値を読み取りグラフを更新する
def graph(*args):
    global ax, cbar
    # axを消去
    ax.cla()
    cbar.remove()
    # スケールバーから値を取得
    _t = scale_var_t.get()
    _p = scale_var_p.get()
    _coil_position = scale_var_coil_position.get()
    _r_up = scale_var_r_up.get()
    _r_down = scale_var_r_down.get()
    value_t = f"{_t:.2f}"
    value_p = f"{_p:.2f}"
    value_coil_position = f"{_coil_position:.2f}"
    value_r_up = f"{_r_up:.2f}"
    value_r_down = f"{_r_down:.2f}"
    text_display_t.set(str(value_t))
    text_display_p.set(str(value_p))
    text_display_coil_position.set(str(value_coil_position))
    text_display_r_up.set(str(value_r_up))
    text_display_r_down.set(str(value_r_down))
    # グラフの描画
    _params = [_t, _p, _coil_position, _r_up, _r_down]
    _T, _U, _V = plotter.predict_all_cell(params=_params, return_prediction=True)
    _im = ax.pcolormesh(X, Y, _T, cmap=cmap, shading="gouraud")
    cbar = fig.colorbar(_im, ax=ax)
    cbar.set_label(label="$T [K]$", rotation=0, labelpad=20)
    ax.set_xlabel("$X [mm]$")
    ax.set_ylabel("$Y [mm]$")
    # 配列要素を2つ飛ばしで描画（::2）
    ax.quiver(X[::2, ::2], Y[::2, ::2], _U[::2, ::2], _V[::2, ::2], cmap="binary_r", width=0.0075)
    plt.style.use("ggplot")
    canvas.draw()


# 各パラメータの設定
t = 2000.0
p = 1000.0
coil_position = -100
r_up = 10
r_down = 60

# モデルを使った計算
params = [t, p, coil_position, r_up, r_down]
plotter = Plotter()
plotter.load_model_and_scaler()
X, Y = plotter.make_grid(0.0, 76.2, 30, 135.0, 239.5, 30, return_grid=True)
T, U, V = plotter.predict_all_cell(params=params, return_prediction=True)

# グラフの描画
fig = plt.Figure(figsize=(7, 7))
ax = fig.add_subplot(111)
cmap = plt.get_cmap("jet")
im = ax.pcolormesh(X, Y, T, cmap=cmap, shading="gouraud")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label="$T [K]$", rotation=0, labelpad=20)
ax.set_xlabel("$X [mm]$")
ax.set_ylabel("$Y [mm]$")
# 配列要素を2つ飛ばしで描画（::2）
ax.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2], cmap="binary_r", width=0.0075)
plt.style.use("ggplot")

# tkinterのウインド上部にグラフを表示する
canvas = FigureCanvasTkAgg(fig, master=frame_1)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# -----温度のパート-----
# 温度のスケール作成
scale_var_t = tk.DoubleVar()
scale_var_t.set(t)
scale_var_t.trace("w", graph)
scale_t = ttk.Scale(frame_2, from_=2000, to=2250, length=150, orient="h", variable=scale_var_t)
scale_t.grid(row=1, column=0)

# 温度のテキスト
text_t = tk.Label(frame_2, text="Temperature [K]")
text_t.grid(row=0, column=0)

# 温度の数値表示テキスト
text_display_t = tk.StringVar()
text_display_t.set(str(t))
label_t = tk.Label(frame_2, textvariable=text_display_t)
label_t.grid(row=1, column=1)

# -----圧力のパート-----
# 圧力のスケール作成
scale_var_p = tk.DoubleVar()
scale_var_p.set(p)
scale_var_p.trace("w", graph)
scale_p = ttk.Scale(frame_2, from_=400, to=2000, length=150, orient="h", variable=scale_var_p)
scale_p.grid(row=3, column=0)

# 圧力のテキスト
text_p = tk.Label(frame_2, text="Pressure [Pa]")
text_p.grid(row=2, column=0)

# 圧力の数値表示テキスト
text_display_p = tk.StringVar()
text_display_p.set(str(p))
label_p = tk.Label(frame_2, textvariable=text_display_p)
label_p.grid(row=3, column=1)

# -----コイル位置のパート-----
# コイル位置のスケール作成
scale_var_coil_position = tk.DoubleVar()
scale_var_coil_position.set(coil_position)
scale_var_coil_position.trace("w", graph)
scale_coil_position = ttk.Scale(frame_2, from_=-100, to=100, length=150, orient="h", variable=scale_var_coil_position)
scale_coil_position.grid(row=5, column=0)

# コイル位置のテキスト
text_coil_position = tk.Label(frame_2, text="Coil position [mm]")
text_coil_position.grid(row=4, column=0)

# コイル位置の数値表示テキスト
text_display_coil_position = tk.StringVar()
text_display_coil_position.set(str(coil_position))
label_coil_position = tk.Label(frame_2, textvariable=text_display_coil_position)
label_coil_position.grid(row=5, column=1)

# -----r_upのパート-----
# r_upのスケール作成
scale_var_r_up = tk.DoubleVar()
scale_var_r_up.set(r_up)
scale_var_r_up.trace("w", graph)
scale_r_up = ttk.Scale(frame_2, from_=10, to=50, length=150, orient="h", variable=scale_var_r_up)
scale_r_up.grid(row=7, column=0)

# r_upのテキスト
text_r_up = tk.Label(frame_2, text="Upper insulation radius[mm]")
text_r_up.grid(row=6, column=0)

# r_upの数値表示テキスト
text_display_r_up = tk.StringVar()
text_display_r_up.set(str(r_up))
label_r_up = tk.Label(frame_2, textvariable=text_display_r_up)
label_r_up.grid(row=7, column=1)

# -----r_downのパート-----
# r_downのスケール作成
scale_var_r_down = tk.DoubleVar()
scale_var_r_down.set(r_down)
scale_var_r_down.trace("w", graph)
scale_r_down = ttk.Scale(frame_2, from_=10, to=60, length=150, orient="h", variable=scale_var_r_down)
scale_r_down.grid(row=9, column=0)

# r_downのテキスト
text_r_down = tk.Label(frame_2, text="Lower insulation radius[mm]")
text_r_down.grid(row=8, column=0)

# r_downの数値表示テキスト
text_display_r_down = tk.StringVar()
text_display_r_down.set(str(r_down))
label_r_down = tk.Label(frame_2, textvariable=text_display_r_down)
label_r_down.grid(row=9, column=1)

root.mainloop()
