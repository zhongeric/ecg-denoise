import os
import numpy as np   
from keras.models import Model 
from keras.layers import Dense, Input  
import matplotlib.pyplot as plt


rootdir = 'E:\\data'
list = os.listdir(rootdir) 
temp_data=[]
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    print(path)
    temp=np.loadtxt(path,dtype=np.float32,delimiter="\t")
    if (len(temp)>10000):
        temp_data.append(temp[:,1][0:10000])
temp_data1 =np.array(temp_data)
print(temp_data1.shape)
temp_data2=temp_data1.reshape(-1,500)
for i in range(temp_data2.shape[0]):
    tempmax=temp_data2[i,:].max(axis=0)
    tempmin=temp_data2[i,:].min(axis=0)
    if (tempmax-tempmin)!=0:
        temp_data2[i,:]=(temp_data2[i,:]-tempmin)/(tempmax-tempmin)
    else:
        temp_data2[i,:]=0.0
    print(temp_data2[i,:].max(axis=0))
# 0---4000 as trainning, 80000--108119 as test
temp_data3=temp_data2[0:40000,:]
print(temp_data3.shape)
print(temp_data2.shape)

input_img = Input(shape=(500,))
 
encoded = Dense(250, activation='relu')(input_img)
encoder_output = Dense(125, activation='relu')(encoded)
 
decoded = Dense(250, activation='relu')(encoder_output)
decoded = Dense(500, activation='tanh')(decoded)
 
autoencoder = Model(inputs=input_img, outputs=decoded)
 
encoder = Model(inputs=input_img, outputs=encoder_output)

#encoded_img = Input(shape=(125,))  
#decoder_layer = autoencoder.layers[-1]  
#decoder = Model(inputs=encoded_img, outputs=decoder_layer(encoded_img)) 

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
 
 # training
autoencoder.fit(temp_data3, temp_data3, epochs=200, batch_size=256, shuffle=True)