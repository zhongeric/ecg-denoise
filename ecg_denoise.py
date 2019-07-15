from keras.datasets import mnist
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Activation, Flatten, Dropout, LeakyReLU, BatchNormalization, GaussianNoise
from keras.models import Model, Sequential, load_model
from keras.optimizers import *
from keras import backend as K
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import math
import time
import os
from numpy import genfromtxt

from progressbar import ProgressBar
pbar = ProgressBar()
rootdir1 = '/home/sxd/eric/QRS-detection/FilterFile'

def read_ecg(file_name):
    return genfromtxt(file_name, delimiter=',')

def baseline_shift(data):
    # get a point in inner quartile range
    max_min_point = random.choice(range(int(data.shape[0] * .25) + 1, int(data.shape[0] * .75)))
    # get number of steps left baseline shift should go
    steps_left = random.choice(np.arange(int(max_min_point) - int(data.shape[0] * .25)))
    steps_right = random.choice(np.arange(int(data.shape[0] * .75) - int(max_min_point)))
    # print(f"Min Max Point For BaseLine Shift: {max_min_point}")
    # print(f"Steps Left For BaseLine Shift: {steps_left}")
    # print(f"Steps Right For BaseLine Shift: {steps_right}")

    transformation_size = steps_left + steps_right
    # print(f"Length of baseline shift: {transformation_size}")

    # average = np.sum(data) / data.shape[0]
    # print(f"Average of the data: {average}") # makes noise too small
    average = 5 #ling factor

    sin_values = np.sin(np.array((0.,30.,45.,60.,90., 120., 135., 160., 180., 210., 225., 240., 270., 300., 315., 330.)) * np.pi / 360.)
    rpcount = math.ceil(transformation_size / len(sin_values))
    large_sin_values = np.repeat(sin_values, rpcount)
    large_sin_values = large_sin_values[0:transformation_size]

    transformed_sin_values = large_sin_values * average

    mm = random.choice(list(range(0,2)))
    if mm == 0:
        # print("Increasing Shift Applied")
        data[(max_min_point - steps_left):(max_min_point + steps_right)] = data[(max_min_point - steps_left):(max_min_point + steps_right)] + transformed_sin_values
    else:
        # print("Decreasing Shift Applied")
        data[(max_min_point - steps_left):(max_min_point + steps_right)] = data[(max_min_point - steps_left):(max_min_point + steps_right)] + (transformed_sin_values * -1)

    return data

# rootdir1 = 'C:\\Users\\cz\\Desktop\\test_for_caizhen\\Twave\\Twave_big'
# rootdir1 = '/home/sxd/eric/QRS-detection/FilterFile_clean'
# data_list = []
# list1 = os.listdir(rootdir1) 
# for i in pbar(range(0,len(list1))):
#     path1 = os.path.join(rootdir1,list1[i])
#     # print(path1)
#     ecg=read_ecg(path1)
#     ecg = ecg[0:1000]
#     data_list.append(ecg)

# np_data = np.asarray(data_list)
# np.savetxt('data.csv', np_data, delimiter=",")

np_data = read_ecg("data.csv")

np_data = np_data[:1000] # just use first 1000 data points plz

split_percent = 0.8

x_train = np_data[0 : int((np_data.shape[0]) * split_percent)]
x_test = np_data[int(np_data.shape[0] * split_percent): ]

print(x_test.shape)

# x_train = x_train.astype("float32") / 255.
# x_test = x_test.astype("float32") / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 5
xtrainn = []
x_train_real = []

for r in pbar(x_train):
    # n = [x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=1000) for x in r]
    # min_val = min(r)
    # u = [(x + abs(min_val)) for x in r]
    n = r + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=r.shape) # randomize amplitude
    d = baseline_shift(n)
    xtrainn.append(d)
    # x_train_real.append(u)

x_test_real = []
xtestn = []
for rr in x_test:
    # min_val = min(rr)
    # u = [(x + abs(min_val)) for x in rr]
    # n = [x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=1000) for x in rr]
    n = rr + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=rr.shape)
    d = baseline_shift(n)
    xtestn.append(d)
    # x_test_real.append(u)


# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# # print(xtrainn)
# x_train = np.asarray(x_train_real)
# x_test = np.asarray(x_test_real)
x_train_noisy = np.asarray(xtrainn)
x_test_noisy = np.asarray(xtestn)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def make_positive(data):
    a = []
    for n in data:
        min_val = min(n)
        u = [(x + abs(min_val)) for x in n]
        a.append(u)
    return a

x_train = normalized(x_train,1)
x_test = normalized(x_test,1)
x_train_noisy = normalized(x_train_noisy,1)
x_test_noisy = normalized(x_test_noisy, 1)


print(f"X Train shape: {x_train.shape}")
print(f"X Test shape: {x_test.shape}")
print(f"X Noisy Train shape: {x_train_noisy.shape}")

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_train_noisy = x_train_noisy.reshape((x_train_noisy.shape[0],x_train_noisy.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
x_test_noisy = x_test_noisy.reshape((x_test_noisy.shape[0],x_test_noisy.shape[1],1))

print(x_train.shape)
input_img = Input(shape=(x_train.shape[1],1))


# x = Conv1D(512, 2, activation="relu", padding="same")(input_img)
# # x = MaxPooling1D(2, padding="same")(x) # 500
# x = Dropout(0.5)(x)
# x = GaussianNoise(0.01)(x)
# x = Conv1D(256, 1, activation="relu", padding="same")(x)
# # x = MaxPooling1D(2, padding="same")(x) #250
# x = Dropout(0.25)(x)
# x = Conv1D(32, 1, activation="linear", padding="same")(x)
# encoded = MaxPooling1D(1, padding="same")(x) #125

# # Representation rn is (7,7,32)

# x = Conv1D(32, 1, activation="relu", padding="same")(encoded)
# # x = UpSampling1D(2)(x)
# x = Dropout(0.5)(x)
# x = Conv1D(256, 1, activation="relu", padding="same")(x)
# # x = UpSampling1D(2)(x)
# x = Dropout(0.25)(x)
# x = Conv1D(512, 2, activation="relu", padding="same")(x)
# x = UpSampling1D(1)(x)
# # x = Dropout(0.25)(x)
# decoded = Conv1D(1, 1, activation="linear", padding='same')(x)

# autoencoder = Model(input_img, decoded)

# print(autoencoder.summary())

# autoencoder.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# autoencoder.fit(x_train_noisy, x_train,
#     epochs = 20,
#     batch_size = 32,
#     shuffle = True,
#     validation_data=(x_test_noisy, x_test))


# autoencoder.save('model23.h5')

autoencoder = load_model('model18.h5')


decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
fs = 250
plt.figure(figsize=(x_train.shape[0], 2))
ecg = np.asarray(random.choice(x_train)) # pull random sample from train
sig_lgth= len(ecg)
ecg=ecg-np.mean(ecg)

index=np.arange(sig_lgth)/fs

x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape[0], x_train_noisy.shape[1])
decoded_imgs = decoded_imgs.reshape(decoded_imgs.shape[0], decoded_imgs.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

fig, ax=plt.subplots()

seed = random.choice(list(range(1,x_test.shape[0])))
print(seed)
ax.set_xlabel('Time [sec]')


# seed = 0 
left_end = 0
right_end = sig_lgth/fs
ind = index
# print(f"INDEX: {ind}")
for i in range(x_test.shape[0]):
    ax.plot(ind, x_test[i], 'k', label='EKG')
    # ax.plot(ind, x_test_noisy[i], 'k', label='Noisy')
    ax.plot(ind, decoded_imgs[i], 'g', label='Decoded')
    ax.set_xlim([left_end, right_end])    
    left_end = right_end
    right_end = left_end + 4
    ind = ind + 4
    plt.pause(5)
        # ax[1].plot(ecg_integrate)
        # ax[1].set_xlim([0, ecg_integrate.shape[0]])
        # ax[2].plot(ecg_lgth_transform)
        # ax[2].set_xlim([0, ecg_lgth_transform.shape[0]])
plt.show()
