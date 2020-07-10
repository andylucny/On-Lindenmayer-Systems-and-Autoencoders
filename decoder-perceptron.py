import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

dataset_size = 1498
    
codes = []
path = 'encoder/rose'
for id in range(dataset_size):
    code = np.loadtxt(path+'{:04d}.txt'.format(id))
    codes.append(code)
    #code = np.loadtxt(path+'{:04d}v.txt'.format(id))
    #codes.append(code)
    #code = np.loadtxt(path+'{:04d}vh.txt'.format(id))
    #codes.append(code)
    #code = np.loadtxt(path+'{:04d}h.txt'.format(id))
    #codes.append(code)
    
codes = np.stack(codes)

params = []
path = 'dataset/rose'
for id in range(dataset_size):
    param = np.loadtxt(path+'{:04d}.txt'.format(id))
    params.append(param)
    #x=param[7]
    #param[7] = 360-x
    #params.append(param)
    #param[7] = 180+x
    #params.append(param)
    #param[7] = 180-x
    #params.append(param)
    
params = np.stack(params)

inp = Input(shape=(8,))
x = Dense(256,activation='tanh')(inp)
x = Dense(256,activation='tanh')(x)
out = Dense(128,activation='sigmoid')(x)
model = Model(inputs=inp, outputs=out)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(params, codes, batch_size=128, epochs=1000)

model.save('decoder-perceptron.h5')


