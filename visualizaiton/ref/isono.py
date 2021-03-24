#!/usr/bin/env python
# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

height = 41
radius = 120
size = height * radius


# data= pd.read_csv('C:/Users/AI-chan/Desktop/ISONO/VAE/data/data_svd_0.5mm.csv')
# no = data[['No.']]
# no = no.values[::size]
# data_2= pd.read_csv('C:/Users/AI-chan/Desktop/ISONO/VAE/data/simulation.csv')
# no_2 = data_2[['No.']]
# diff_num = np.setdiff1d(np.ravel(no), np.ravel(no_2))
# for i in diff_num:
#     data = data.drop(data.index[data["No."] == i])
# all_data = pd.read_csv('C:/Users/AI-chan/Desktop/ISONO/VAE/data/save_data.csv')
# all_data=all_data.drop(all_data['No.'] == 3)
# process = all_data[['rot_up','rot_low','crucible_position','x','y']]
# process = process.values
# process = process.reshape(-1,size,5)
# process = process.values[::size]


def data_read(path):
    csv_list = glob.glob(path)
    train_data = np.empty((0, 9))
    for csv_name in csv_list:
        data = np.loadtxt(csv_name, delimiter=",", skiprows=1,
                          usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8), dtype="float")
        print(csv_name)
        print(data.shape)
        train_data = np.vstack((train_data, data))
    return train_data


train_data = data_read('/Users/isonomasaru/Downloads/save_data/train_data/*.csv')
valid_data = data_read('/Users/isonomasaru/Downloads/save_data/valid_data/*.csv')
test_data = data_read('/Users/isonomasaru/Downloads/save_data/test_data/*.csv')

p_train = train_data[:, 1:6].reshape(-1, size, 5)
p_valid = valid_data[:, 1:6].reshape(-1, size, 5)
p_test = test_data[:, 1:6].reshape(-1, size, 5)

x_train = train_data[:, 6:].reshape(-1, size, 3)
x_valid = valid_data[:, 6:].reshape(-1, size, 3)
x_test = test_data[:, 6:].reshape(-1, size, 3)

# T = all_data[['T']]
# T = T.values
# Vx = all_data[['Vx']]
# Vx = Vx.values
# Vy = all_data[['Vy']]
# Vy = Vy.values

# T = T.reshape(-1,1)
# Vx = Vx.reshape(-1,1)
# Vy = Vy.reshape(-1,1)

# x_train = np.hstack((T,Vx,Vy))
# p_train,p_test,x_train, x_test = train_test_split(process,x_train,test_size=0.2,random_state=50)
# p_train,p_valid,x_train, x_valid = train_test_split(p_train,x_train,test_size=0.1,random_state=50)
p_train = p_train.reshape(-1, 5)
p_valid = p_valid.reshape(-1, 5)
p_test = p_test.reshape(-1, 5)
x_train = x_train.reshape(-1, 3)
x_valid = x_valid.reshape(-1, 3)
x_test = x_test.reshape(-1, 3)
x_train = x_train[:, 0].reshape(-1, 1)
x_valid = x_valid[:, 0].reshape(-1, 1)
x_test = x_test[:, 0].reshape(-1, 1)

p_stdsc = StandardScaler()
x_stdsc = StandardScaler()
p_train_std = p_stdsc.fit_transform(p_train)
p_valid_std = p_stdsc.transform(p_valid)
p_test_std = p_stdsc.transform(p_test)
x_train_std = x_stdsc.fit_transform(x_train)
x_valid_std = x_stdsc.transform(x_valid)
x_test_std = x_stdsc.transform(x_test)
print(x_test.shape)
print(x_train_std)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2
regularizer = regularizers.l2(1e-4)

encoder_inputs = keras.Input(shape=(41, 120, 1))
# x = layers.Dropout(0.2)(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(x)
x = layers.Dropout(0.5)(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(latent_inputs)
x = layers.Dense(6 * 15 * 64, activation="relu", kernel_regularizer=regularizer)(x)
x = layers.Reshape((6, 15, 64))(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_regularizer=regularizer)(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(x)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Conv2D(1, 3, activation="linear", padding="same", kernel_regularizer=regularizer)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mse(data, reconstruction)
            )
            reconstruction_loss *= 41 * 120 * 1
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(data, reconstruction)
        )
        reconstruction_loss *= 41 * 120 * 1
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


# T,Vx,Vyまとめたモデル
x_train = x_train.reshape(-1, 1)
x_valid = x_valid.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
vae_stdsc = StandardScaler()
vae_train_std = vae_stdsc.fit_transform(x_train).reshape(-1, 41, 120, 1)
vae_valid_std = vae_stdsc.transform(x_valid).reshape(-1, 41, 120, 1)
vae_test_std = vae_stdsc.transform(x_test).reshape(-1, 41, 120, 1)

original_dim = size * 1
result_dir = '/Users/isonomasaru/Desktop/cnn_vae_T'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)  # もし指定フォルダがなかったら作成する

modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath=result_dir + '/VAE_best_weights.h5',
                                                  monitor='val_loss',
                                                  verbose=0,
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  mode='min')


def step_decay(epoch):
    x = 1e-3
    if epoch >= 100:
        x = 1e-4
    if epoch >= 160:
        x = 1e-5
    return x


lr_decay = keras.callbacks.LearningRateScheduler(step_decay, verbose=0)
vae = VAE(encoder, decoder)

weights = result_dir + '/encoder_weights.h5'
weights_1 = result_dir + '/decoder_weights.h5'

# weights=False
if weights:
    encoder.load_weights(weights)
    decoder.load_weights(weights_1)
else:
    encoder_str = encoder.to_json()
    open(result_dir + '/encoder_model.json', 'w').write(encoder_str)
    decoder_str = decoder.to_json()
    open(result_dir + '/decoder_model.json', 'w').write(decoder_str)
    # VAE_str = vae.to_json()
    # open(result_dir+'/VAE_model.json', 'w').write(VAE_str)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    fit = vae.fit(vae_train_std, epochs=2000, batch_size=32, validation_data=(vae_valid_std, vae_valid_std))


    def plot_history_loss(fit):
        # Plot the loss in the history
        plt.plot(fit.history['loss'], label="loss for training")
        plt.plot(fit.history['val_loss'], label="loss for validation")
        plt.legend(loc='upper right')
        plt.yscale('log')


    plot_history_loss(fit)
    save = True
    if save == True:
        plt.savefig(result_dir + '/loss.png')
        # VAE_str = vae.to_json()
        # open(result_dir+'/VAE_model.json', 'w').write(VAE_str)
        encoder.save_weights(result_dir + '/encoder_weights.h5')
        decoder.save_weights(result_dir + '/decoder_weights.h5')
    plt.close()


def vae_predict(vae_test_std):
    z_mean, z_log_var, z = encoder.predict(vae_test_std)
    decoder_pred_std = decoder.predict(z)
    return decoder_pred_std


vae_pred_std = vae_predict(vae_train_std)
vae_pred = vae_stdsc.inverse_transform(vae_pred_std.reshape(-1, size * 1))
rmse = np.sqrt(mean_squared_error(vae_pred.reshape(-1, 41, 120, 1)[:, :, :, 0].flatten(),
                                  x_train.reshape(-1, 41, 120, 1)[:, :, :, 0].flatten()))
print(rmse)
r2 = r2_score(vae_train_std.flatten(), vae_pred_std.flatten())
print('R2:' + str(r2))


def yyplot(y_obs, y_pred):
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues) - 0.1, np.amax(yvalues) + 0.1, np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.3
    plt.scatter(y_obs, y_pred, s=20, alpha=0.7)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('input', fontsize=24)
    plt.ylabel('pred', fontsize=24)
    plt.tick_params(labelsize=30)
    plt.show()
    return fig


fig = yyplot(vae_pred.reshape(-1, 41, 120, 1), x_train.reshape(-1, 41, 120, 1))


# fig = plt.figure(figsize=(13.5, 16))
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# X=np.arange(120)
# Y=np.arange(41)
# ax.set_xlim(0, 119)
# ax.set_ylim(0,40)

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# ax_im = ax.imshow(vae_train_std.reshape(-1,41,120,3)[0,:,:,0],cmap='jet')
# ax.streamplot(X, Y, vae_train_std.reshape(-1,41,120,3)[0,:,:,1], vae_train_std.reshape(-1,41,120,3)[0,:,:,2], linewidth=2,color="black",arrowsize = 1,density=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', '3%', '2%')
# cbar = fig.colorbar(ax_im,cax = cax)
# cbar.ax.tick_params(labelsize=20)
# ax.tick_params(bottom=False,
#                     left=False,
#                     right=False,
#                     top=False,
#                     labelbottom=False,
#                     labelleft=False,
#                     labelright=False,
#                     labeltop=False)
# plt.show()

# fig = plt.figure(figsize=(13.5, 16))
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')
# X=np.arange(120)
# Y=np.arange(41)
# ax.set_xlim(0, 119)
# ax.set_ylim(0,40)

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# ax_im = ax.imshow(vae_pred_std.reshape(-1,41,120,3)[0,:,:,0],cmap='jet')
# ax.streamplot(X, Y, vae_pred_std.reshape(-1,41,120,3)[0,:,:,1], vae_pred_std.reshape(-1,41,120,3)[0,:,:,2], linewidth=2,color="black",arrowsize = 1,density=2)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', '3%', '2%')
# cbar = fig.colorbar(ax_im,cax = cax)
# cbar.ax.tick_params(labelsize=20)
# ax.tick_params(bottom=False,
#                     left=False,
#                     right=False,
#                     top=False,
#                     labelbottom=False,
#                     labelleft=False,
#                     labelright=False,
#                     labeltop=False)
# plt.show()


def vae_predict(vae_test_std):
    z_mean, z_log_var, z = encoder.predict(vae_test_std)
    decoder_pred_std = decoder.predict(z)
    return decoder_pred_std


vae_pred_std = vae_predict(vae_test_std).reshape(-1, 1)
vae_pred = vae_stdsc.inverse_transform(vae_pred_std)
rmse = np.sqrt(mean_squared_error(vae_pred.reshape(-1, 41, 120, 1)[:, :, :, 0].flatten(),
                                  x_test.reshape(-1, 41, 120, 1)[:, :, :, 0].flatten()))
print(rmse)
r2 = r2_score(vae_test_std.reshape(-1, 1)[:, 0].flatten(), vae_pred_std[:, 0].flatten())
print('R2:' + str(r2))


def yyplot(y_obs, y_pred):
    yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
    ymin, ymax, yrange = np.amin(yvalues) - 0.1, np.amax(yvalues) + 0.1, np.ptp(yvalues)
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'sans-serif'

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.3
    plt.scatter(y_obs, y_pred, s=10, alpha=0.3)
    plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
    plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
    plt.xlabel('input', fontsize=24)
    plt.ylabel('pred', fontsize=24)
    plt.tick_params(labelsize=30)
    plt.show()
    return fig


vae_pred = vae_pred.reshape(-1, 41, 120, 1)
fig = yyplot(vae_pred.reshape(-1, 41, 120, 1)[:, :, :, 0], x_test.reshape(-1, 41, 120, 1)[:, :, :, 0])
n = 2
fig = plt.figure(figsize=(13.5, 16))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
X = np.arange(120)
Y = np.arange(41)
ax.set_xlim(0, 119)
ax.set_ylim(0, 40)
decoded_img = decoded_img.reshape(-1, 41, 120, 1)

from mpl_toolkits.axes_grid1 import make_axes_locatable

# V_mag = np.sqrt((all_train[1,:,:,1])**2+(all_train[0,:,:,2])**2)
ax_im = ax.imshow(vae_pred[n, :, :, 0], cmap='jet', vmin=2188, vmax=2205)
# ax.streamplot(X, Y, all_train[0,:,:,1], all_train[0,:,:,2], linewidth=3,color="gray",arrowsize = 2,density=2)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', '3%', '2%')
cbar = fig.colorbar(ax_im, cax=cax, ticks=np.arange(2188, 2205, 4))
cbar.ax.tick_params(labelsize=30)
ax.tick_params(bottom=False,
               left=False,
               right=False,
               top=False,
               labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
plt.show()
plt.close()

z_mean, z_log_var, z = encoder.predict(vae_train_std)
print(z.shape)
plt.figure()
plt.scatter(z[:, 0], z[:, 1], s=20)
plt.tick_params(labelsize=20)
plt.xlabel("Feature 1", fontsize=24)
plt.ylabel("Feature 2", fontsize=24)
plt.show()

method_dict = {1: 'Random Forest', 2: 'Support Vector Machine', 3: 'Neural Network'}
number = 1
method = method_dict[number]
print(method)
process_train = p_train[::size, :3]
process_valid = p_valid[::size, :3]
process_test = p_test[::size, :3]
process_train_std = p_train_std[::size, :3]
process_valid_std = p_valid_std[::size, :3]
process_test_std = p_test_std[::size, :3]


def mean_var(vae_std):
    vae_std = vae_std.reshape(-1, 41, 120, 1)
    mean_var = encoder.predict(vae_std)
    mean = np.array(mean_var[0])
    var = np.array(mean_var[1])
    mean_var = np.c_[mean, var]
    return mean_var


mean_var_train = mean_var(vae_train_std)
mean_var_valid = mean_var(vae_valid_std)
mean_var_test = mean_var(vae_test_std)

latent_std = preprocessing.StandardScaler()
mean_var_train_std = latent_std.fit_transform(mean_var_train)
mean_var_valid_std = latent_std.transform(mean_var_valid)
mean_var_test_std = latent_std.transform(mean_var_test)

if method == method_dict[1]:
    rg = RFR(random_state=100, n_estimators=100)
    rg.fit(process_train, mean_var_train)
    mean_var_pred = rg.predict(process_test)


    def latent_pred(process):
        latent = rg.predict(process)
        return latent

elif method == method_dict[2]:
    latent_std = preprocessing.StandardScaler()
    latent_tr = latent_std.fit_transform(latent_tr)
    latent_ts = latent_std.transform(latent_ts)
    process_std = preprocessing.StandardScaler()
    process_tr = process_std.fit_transform(process_tr)
    process_ts = process_std.transform(process_ts)
    all_pred = np.empty((8, 0))

    C_list = [3.1218245793152035, 80029997.61202382, 22222345.403845366, 5.873189524308266]
    eps_list = [2.2711076385477055e-05, 0.08626100011777312, 4.345446835724542e-10, 2.9234247274671955e-05, ]
    gamma_list = [0.2395525029124204, 1.28784194074448, 7.057926268795237e-09, 0.1846938846331795]
    for (i, C, eps, gamma) in zip(range(4), C_list, eps_list, gamma_list):
        print(C, gamma)
        kernel = 'rbf'
        exec('reg_' + str(i) + '= svm.SVR(kernel=kernel,C=C,epsilon = eps, gamma=gamma)')
        exec('reg_' + str(i) + '.fit(process_tr, latent_tr[:,' + str(i) + '])')
        exec('pred = reg_' + str(i) + '.predict(process_ts)')
        pred = pred.reshape(-1, 1)
        all_pred = np.hstack((all_pred, pred))
    latent_ts = latent_std.inverse_transform(latent_ts)
    print(latent_ts)
    all_pred = latent_std.inverse_transform(all_pred)
    print(all_pred)
    print(np.sqrt(mean_squared_error(latent_ts, all_pred)))
    r2 = r2_score(latent_ts, all_pred)
    print(r2)

    model = model_from_json(open("/Users/isonomasaru/Desktop/nn_weight/nn_model.json").read())
    weights = "/Users/isonomasaru/Desktop/nn_weight/nn_weights.h5"
    model.load_weights(weights)
    pred = model.predict(process_ts)
    latent_ts = latent_std.inverse_transform(latent_ts)
    pred = latent_std.inverse_transform(pred)
    print(np.sqrt(mean_squared_error(latent_ts, pred)))


    def latent_pred(process):
        process = process_std.transform(process)
        all_pred = np.empty((len(process), 0))
        pred = reg_0.predict(process)
        pred = pred.reshape(-1, 1)
        all_pred = np.hstack((all_pred, pred))
        pred = reg_1.predict(process)
        pred = pred.reshape(-1, 1)
        all_pred = np.hstack((all_pred, pred))
        pred = reg_2.predict(process)
        pred = pred.reshape(-1, 1)
        all_pred = np.hstack((all_pred, pred))
        pred = reg_3.predict(process)
        pred = pred.reshape(-1, 1)
        all_pred = np.hstack((all_pred, pred))
        latent = latent_std.inverse_transform(all_pred)
        return latent

elif method == method_dict[3]:
    from keras.layers import Dense, BatchNormalization, Activation

    process_train = p_train[::size, :3]
    process_valid = p_valid[::size, :3]
    process_test = p_test[::size, :3]
    test_std = preprocessing.MinMaxScaler()
    process_train_mm = test_std.fit_transform(process_train)
    process_valid_mm = test_std.transform(process_valid)
    process_test_mm = test_std.transform(process_test)
    inputs = layers.Input(shape=(3,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    outputs = layers.Dense(2, activation='linear')(x)

    # this model maps an input to its reconstruction
    neural_network = keras.Model(inputs, outputs)
    neural_network.summary()
    neural_network.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=1e-4))
    history = neural_network.fit(process_train_mm, mean_var_train_std, epochs=2000, batch_size=32,
                                 validation_data=(process_valid_mm, mean_var_valid_std))
    mean_var_pred_std = neural_network.predict(process_test_mm)
    mean_var_pred = latent_std.inverse_transform(mean_var_pred_std)


    def latent_pred(process):
        latent_std = neural_network.predict(process)
        latent = latent_std.inverse_transform(latent_std)
        return latent
else:
    pass

# mean_var_test = mean_var_train
fig = yyplot(mean_var_test[:, :2], mean_var_pred[:, :2])
plt.figure()
plt.scatter(mean_var_test[:, 0], mean_var_test[:, 1], c='b')
plt.scatter(mean_var_pred[:, 0], mean_var_pred[:, 1], c='r')
plt.show()
plt.close()
rmse = np.sqrt(mean_squared_error(mean_var_test[:, :2].flatten(), mean_var_pred[:, :2].flatten()))
print('RMSE:' + str(rmse))
r2 = r2_score(mean_var_test[:, :2].flatten(), mean_var_pred[:, :2].flatten())
print('R2:' + str(r2))

z_mean, z_log_var, _ = encoder.predict(vae_train_std)


def latent_sampling(mean, log_var):
    batch = np.shape(mean)[0]
    dim = np.shape(mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = np.random.normal(0, 1, (batch, dim))
    return mean + np.exp(0.5 * log_var) * epsilon


# ここからGUI作成部分
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter import ttk
import mpl_toolkits.axes_grid1

root = tk.Tk()  # ウィンドウの作成
root.title("process_change_trajectory")  # ウィンドウのタイトル
root.geometry("800x400")  # ウインドの大きさ
nb = ttk.Notebook(width=600, height=400)

# タブの作成
tab1 = tk.Frame(nb)
tab2 = tk.Frame(nb)
tab3 = tk.Frame(nb)
nb.add(tab1, text='rot_up', padding=3)
nb.add(tab2, text='rot_low', padding=3)
nb.add(tab3, text='crucible_position', padding=3)
nb.pack(expand=1, fill='both')

frame_1_1 = tk.LabelFrame(tab1, labelanchor="nw", text="グラフ", foreground="green")
frame_1_1.grid(row=0, column=0)
frame_2_1 = tk.LabelFrame(tab1, labelanchor="nw", text="パラメータ", foreground="green")
frame_2_1.grid(row=0, column=1, sticky="nwse")

frame_1_2 = tk.LabelFrame(tab2, labelanchor="nw", text="グラフ", foreground="green")
frame_1_2.grid(row=0, column=0)
frame_2_2 = tk.LabelFrame(tab2, labelanchor="nw", text="パラメータ", foreground="green")
frame_2_2.grid(row=0, column=1, sticky="nwse")

frame_1_3 = tk.LabelFrame(tab3, labelanchor="nw", text="グラフ", foreground="green")
frame_1_3.grid(row=0, column=0)
frame_2_3 = tk.LabelFrame(tab3, labelanchor="nw", text="パラメータ", foreground="green")
frame_2_3.grid(row=0, column=1, sticky="nwse")

plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.3


# -----rot_up-----
# rot_upが変化したら、値を読み込みグラフを更新する
def rot_up_change(*args):
    ax.cla()
    ax.set_aspect('equal')
    rot_low = scale_var_rot_low.get()
    crucible_position = scale_var_crucible_position.get() + 216.52
    rot_up = np.arange(0, 101, 1).reshape(-1, 1)
    rot_low_1 = (np.zeros((len(rot_up),)) + rot_low).reshape(-1, 1)
    crucible_position_1 = (np.zeros((len(rot_up),)) + crucible_position).reshape(-1, 1)
    process = np.hstack((rot_up, rot_low_1, crucible_position_1))
    color = process[:, 0]
    pr = latent_pred(process)
    z_mean_1 = pr[:, 0:2]
    z_log_var_1 = pr[:, 2:4]
    z = latent_sampling(z_mean_1, z_log_var_1)
    ax.scatter(z_mean[:, 0], z_mean[:, 1], s=120, c='w')
    ax.scatter(z[:, 0], z[:, 1], s=120, c=color, cmap='jet')
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Feature 1", fontsize=24)
    ax.set_ylabel("Feature 2", fontsize=24)
    canvas.draw()


rot_low = 0
crucible_position = 466.52
rot_up = np.arange(0, 101, 1).reshape(-1, 1)
rot_low_1 = (np.zeros((len(rot_up),)) + rot_low).reshape(-1, 1)
crucible_position_1 = (np.zeros((len(rot_up),)) + crucible_position).reshape(-1, 1)
process = np.hstack((rot_up, rot_low_1, crucible_position_1))
color = process[:, 0]
pr = latent_pred(process)
z_mean_1 = pr[:, 0:2]
z_log_var_1 = pr[:, 2:4]
z = latent_sampling(z_mean_1, z_log_var_1)
# グラフの設定
fig = plt.Figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.scatter(z_mean[:, 0], z_mean[:, 1], s=120)
ax_fig = ax.scatter(z[:, 0], z[:, 1], s=120, c=color, cmap='jet')
ax.set_title('rot_low=' + str(rot_low) + ',crucible_position=' + str(crucible_position - 216.52))
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
cax = divider.append_axes('right', '5%', pad='3%')
cbar0 = fig.colorbar(ax_fig, cax=cax)
cbar0.ax.tick_params(labelsize=20)

# tkinterのウインド上部にグラフを表示する
canvas = FigureCanvasTkAgg(fig, master=frame_1_1)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# スケールの作成
scale_var_rot_low = tk.DoubleVar()
scale_var_rot_low.set(rot_low)
scale_var_rot_low.trace("w", rot_up_change)
scale_rot_low = tk.Scale(frame_2_1, from_=-100, to=100, length=150, orient="h", variable=scale_var_rot_low)
scale_rot_low.grid(row=3, column=0)

# 表示用のテキスト
text_rot_low = tk.Label(frame_2_1, text="rot_low")
text_rot_low.grid(row=2, column=0)

# スケールの作成
scale_var_crucible_position = tk.DoubleVar()
scale_var_crucible_position.set(crucible_position - 216.52)
scale_var_crucible_position.trace("w", rot_up_change)
scale_crucible_position = tk.Scale(frame_2_1, from_=250, to=300, length=150, orient="h",
                                   variable=scale_var_crucible_position)
scale_crucible_position.grid(row=5, column=0)

# 表示用のテキスト
text_crucible_position = tk.Label(frame_2_1, text="crucible_position")
text_crucible_position.grid(row=4, column=0)


# -----rot_low-----
# rot_lowが変化したら、値を読み込みグラフを更新する
def rot_low_change(*args):
    ax_1.cla()
    ax_1.set_aspect('equal')
    rot_up_1 = scale_var_rot_up_1.get()
    crucible_position_1 = scale_var_crucible_position_1.get() + 216.52
    rot_low_1 = np.arange(-100, 101, 1).reshape(-1, 1)
    rot_up_1_1 = (np.zeros((len(rot_low_1),)) + rot_up_1).reshape(-1, 1)
    crucible_position_1_1 = (np.zeros((len(rot_low_1),)) + crucible_position_1).reshape(-1, 1)
    process_1 = np.hstack((rot_up_1_1, rot_low_1, crucible_position_1_1))
    color_1 = process_1[:, 1]
    pr_2 = latent_pred(process_1)
    z_mean_1_1 = pr_2[:, 0:2]
    z_log_var_1_1 = pr_2[:, 2:4]
    z_1 = latent_sampling(z_mean_1_1, z_log_var_1_1)
    ax_1.scatter(z_mean[:, 0], z_mean[:, 1], s=120, c='w')
    ax_1.scatter(z_1[:, 0], z_1[:, 1], s=120, c=color_1, cmap='jet')
    ax_1.tick_params(labelsize=20)
    ax_1.set_xlabel("Feature 1", fontsize=24)
    ax_1.set_ylabel("Feature 2", fontsize=24)
    plt.tight_layout()
    canvas_1.draw()


rot_up_1 = 0
crucible_position_1 = 466.52
rot_low_1 = np.arange(-100, 101, 1).reshape(-1, 1)
rot_up_1_1 = (np.zeros((len(rot_low_1),)) + rot_up_1).reshape(-1, 1)
crucible_position_1_1 = (np.zeros((len(rot_low_1),)) + crucible_position_1).reshape(-1, 1)
process_1 = np.hstack((rot_up_1_1, rot_low_1, crucible_position_1_1))
color_1 = process_1[:, 1]
pr_2 = latent_pred(process_1)
z_mean_1_1 = pr_2[:, 0:2]
z_log_var_1_1 = pr_2[:, 2:4]
z_1 = latent_sampling(z_mean_1_1, z_log_var_1_1)
# グラフの設定
fig_1 = plt.Figure(figsize=(7, 7))
ax_1 = fig_1.add_subplot(111)
ax_1.set_aspect('equal')
ax_1.scatter(z_mean[:, 0], z_mean[:, 1], s=120, c='w')
ax_1_fig = ax_1.scatter(z_1[:, 0], z_1[:, 1], s=120, c=color_1, cmap='jet')

ax_1.set_title('rot_up=' + str(rot_up_1) + ',crucible_position=' + str(crucible_position_1 - 216.52))
ax_1.tick_params(labelsize=20)
ax_1.set_xlabel("Feature 1", fontsize=24)
ax_1.set_ylabel("Feature 2", fontsize=24)
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax_1)
cax = divider.append_axes('right', '5%', pad='3%')
cbar = fig_1.colorbar(ax_1_fig, cax=cax)
cbar.ax.tick_params(labelsize=20)

# tkinterのウインド上部にグラフを表示する
canvas_1 = FigureCanvasTkAgg(fig_1, master=frame_1_2)
canvas_1.draw()
canvas_1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# スケールの作成
scale_var_rot_up_1 = tk.DoubleVar()
scale_var_rot_up_1.set(rot_up_1)
scale_var_rot_up_1.trace("w", rot_low_change)
scale_rot_up_1 = tk.Scale(frame_2_2, from_=0, to=100, length=150, orient="h", variable=scale_var_rot_up_1)
scale_rot_up_1.grid(row=3, column=0)

# 表示用のテキスト
text_rot_up_1 = tk.Label(frame_2_2, text="rot_up")
text_rot_up_1.grid(row=2, column=0)

# スケールの作成
scale_var_crucible_position_1 = tk.DoubleVar()
scale_var_crucible_position_1.set(crucible_position_1 - 216.52)
scale_var_crucible_position_1.trace("w", rot_low_change)
scale_crucible_position_1 = tk.Scale(frame_2_2, from_=250, to=300, length=150, orient="h",
                                     variable=scale_var_crucible_position_1)
scale_crucible_position_1.grid(row=5, column=0)

# 表示用のテキスト
text_crucible_position_1 = tk.Label(frame_2_2, text="crucible_position")
text_crucible_position_1.grid(row=4, column=0)


# -----crucible_position-----
# crucible_positionが変化したら、値を読み込みグラフを更新する
def crucible_position_change(*args):
    ax_2.cla()
    ax_2.set_aspect('equal')
    rot_low_2 = scale_var_rot_low_2.get()
    rot_up_2 = scale_var_rot_up_2.get()
    crucible_position_2 = np.arange(466.52, 517.52, 1).reshape(-1, 1)
    rot_low_2_1 = (np.zeros((len(crucible_position_2),)) + rot_low_2).reshape(-1, 1)
    rot_up_2_1 = (np.zeros((len(crucible_position_2),)) + rot_up_2).reshape(-1, 1)
    process_2 = np.hstack((rot_up_2_1, rot_low_2_1, crucible_position_2))
    color_2 = process_2[:, 2] - 216.52
    pr_2 = latent_pred(process_2)
    z_mean_2_1 = pr_2[:, 0:2]
    z_log_var_2_1 = pr_2[:, 2:4]
    z_2 = latent_sampling(z_mean_2_1, z_log_var_2_1)
    ax_2.scatter(z_mean[:, 0], z_mean[:, 1], s=120, c='w')
    ax_2.scatter(z_2[:, 0], z_2[:, 1], s=120, c=color_2, cmap='jet')
    ax_2.tick_params(labelsize=20)
    ax_2.set_xlabel("Feature 1", fontsize=24)
    ax_2.set_ylabel("Feature 2", fontsize=24)
    canvas_2.draw()


rot_low_2 = 0
rot_up_2 = 0
crucible_position_2 = np.arange(466.52, 517.52, 1).reshape(-1, 1)
rot_low_2_1 = (np.zeros((len(crucible_position_2),)) + rot_low_2).reshape(-1, 1)
rot_up_2_1 = (np.zeros((len(crucible_position_2),)) + rot_up_2).reshape(-1, 1)
process_2 = np.hstack((rot_up_2_1, rot_low_2_1, crucible_position_2))
color_2 = process_2[:, 2] - 216.52
pr_2 = latent_pred(process_2)
z_mean_2_1 = pr_2[:, 0:2]
z_log_var_2_1 = pr_2[:, 2:4]
z_2 = latent_sampling(z_mean_2_1, z_log_var_2_1)
# グラフの設定
fig_2 = plt.Figure(figsize=(7, 7))
ax_2 = fig_2.add_subplot(111)
ax_2.set_aspect('equal')
ax_2.scatter(z_mean[:, 0], z_mean[:, 1], s=120, c='w')
ax_2_fig = ax_2.scatter(z_2[:, 0], z_2[:, 1], s=120, c=color_2, cmap='jet')
ax_2.set_title('rot_up=' + str(rot_up_2) + ',rot_low=' + str(rot_low_2))
ax_2.set_xlabel("Feature 1")
ax_2.set_ylabel("Feature 2")
divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax_2)
cax = divider.append_axes('right', '5%', pad='3%')
cbar = fig_2.colorbar(ax_2_fig, cax=cax)
cbar.ax.tick_params(labelsize=20)

# tkinterのウインド上部にグラフを表示する
canvas_2 = FigureCanvasTkAgg(fig_2, master=frame_1_3)
canvas_2.draw()
canvas_2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# スケールの作成
scale_var_rot_up_2 = tk.DoubleVar()
scale_var_rot_up_2.set(rot_up_2)
scale_var_rot_up_2.trace("w", crucible_position_change)
scale_rot_up_2 = tk.Scale(frame_2_3, from_=0, to=100, length=150, orient="h", variable=scale_var_rot_up_2)
scale_rot_up_2.grid(row=3, column=0)

# 表示用のテキスト
text_rot_up_2 = tk.Label(frame_2_3, text="rot_up")
text_rot_up_2.grid(row=2, column=0)

# スケールの作成
scale_var_rot_low_2 = tk.DoubleVar()
scale_var_rot_low_2.set(rot_low_2)
scale_var_rot_low_2.trace("w", crucible_position_change)
scale_rot_low_2 = tk.Scale(frame_2_3, from_=-100, to=100, length=150, orient="h", variable=scale_var_rot_low_2)
scale_rot_low_2.grid(row=5, column=0)

# 表示用のテキスト
text_rot_low_2 = tk.Label(frame_2_3, text="rot_low")
text_rot_low_2.grid(row=4, column=0)

root.mainloop()
# ## ここまで
