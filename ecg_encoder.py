from keras.datasets import mnist
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Activation, Flatten, Dropout, LeakyReLU
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os
from numpy import genfromtxt

from progressbar import ProgressBar
pbar = ProgressBar()
rootdir1 = '/home/sxd/eric/QRS-detection/FilterFile'

def read_ecg(file_name):
    return genfromtxt(file_name, delimiter=',')

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

noise_factor = 3
xtrainn = []
for r in pbar(x_train):
    # n = [x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=1000) for x in r]
    n = r + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=r.shape)
    xtrainn.append(n)

xtestn = []
for rr in x_test:
    # n = [x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=1000) for x in rr]
    n = rr + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=rr.shape)
    xtestn.append(n)


# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# print(xtrainn)

x_train_noisy = np.asarray(xtrainn)
x_test_noisy = np.asarray(xtestn)


print(f"X Train shape: {x_train.shape}")
print(f"X Test shape: {x_test.shape}")
print(f"X Noisy Train shape: {x_train_noisy.shape}")

# n = 10
# fs = 250
# plt.figure(figsize=(x_train.shape[0], 2))

# ecg = np.asarray(random.choice(x_train[0])) # pull random sample from train
# sig_lgth=ecg.shape[0]
# ecg=ecg-np.mean(ecg)

# index=np.arange(sig_lgth)/fs

# index = index[0 : int(len(index) * split_percent)]

# fig, ax=plt.subplots()
# ax.plot(index, x_train, 'b', label='EKG')
# ax.plot(index, x_train_noisy, 'r', label='Noisy EKG')
# ax.set_xlim([0, sig_lgth/fs])
# ax.set_xlabel('Time [sec]')
#         # ax[1].plot(ecg_integrate)
#         # ax[1].set_xlim([0, ecg_integrate.shape[0]])
#         # ax[2].plot(ecg_lgth_transform)
#         # ax[2].set_xlim([0, ecg_lgth_transform.shape[0]])
# plt.show()
# print(x_train.shape[0])

# model = Sequential()
# model.add(Dense(1000, input_shape=(x_train.shape[1],)))
# model.add(Activation('relu'))
# # model.add(Flatten())
# model.compile(optimizer='adadelta', loss='binary_crossentropy')


# x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
# x_train_noisy = x_train_noisy.reshape((x_train_noisy.shape[0],x_train_noisy.shape[1],1))
# x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
# x_test_noisy = x_test_noisy.reshape((x_test_noisy.shape[0],x_test_noisy.shape[1],1))

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]))
x_train_noisy = x_train_noisy.reshape((x_train_noisy.shape[0],x_train_noisy.shape[1]))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]))
x_test_noisy = x_test_noisy.reshape((x_test_noisy.shape[0],x_test_noisy.shape[1]))

print(x_train.shape)
input_img = Input(shape=(x_train.shape[1],))

encoded = Dense(250, activation='relu')(input_img)
encoded = Dropout(0.5)(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(56, activation='relu')(encoded)
encoder_output = Dense(125, activation='relu')(encoded)

decoded = Dense(250, activation='relu')(encoder_output)
decoded = Dropout(0.5)(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(1000, activation='linear')(decoded)
 
autoencoder = Model(inputs=input_img, outputs=decoded)
 
encoder = Model(inputs=input_img, outputs=encoder_output)

#encoded_img = Input(shape=(125,))  
#decoder_layer = autoencoder.layers[-1]  
#decoder = Model(inputs=encoded_img, outputs=decoder_layer(encoded_img)) 

# compile autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
 
 # training

# autoencoder.fit(temp_data3, temp_data3, epochs=200, batch_size=256, shuffle=True)

autoencoder.fit(x_train_noisy, x_train,
    epochs = 50,
    batch_size = 32,
    shuffle = True,
    validation_data=(x_test, x_test_noisy))

print(autoencoder.summary())
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
ax.plot(index, x_test[0], 'k', label='EKG')
# ax.plot(index, x_test_noisy[0], 'b', label='EKG')
ax.plot(index, decoded_imgs[0], 'r', label='Noisy EKG')
ax.set_xlim([0, sig_lgth/fs])
ax.set_xlabel('Time [sec]')
        # ax[1].plot(ecg_integrate)
        # ax[1].set_xlim([0, ecg_integrate.shape[0]])
        # ax[2].plot(ecg_lgth_transform)
        # ax[2].set_xlim([0, ecg_lgth_transform.shape[0]])
plt.show()
