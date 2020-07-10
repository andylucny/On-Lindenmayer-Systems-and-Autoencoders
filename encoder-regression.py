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
    params.append([np.cos(np.pi*param[0]/180.0),np.sin(np.pi*param[0]/180.0),np.cos(np.pi*param[7]/180.0),np.sin(np.pi*param[7]/180.0)])
    #x=param[7]
    #param[7] = 360-x
    #params.append([np.cos(np.pi*param[0]/180.0),np.sin(np.pi*param[0]/180.0),np.cos(np.pi*param[7]/180.0),np.sin(np.pi*param[7]/180.0)])
    #param[7] = 180+x
    #params.append([np.cos(np.pi*param[0]/180.0),np.sin(np.pi*param[0]/180.0),np.cos(np.pi*param[7]/180.0),np.sin(np.pi*param[7]/180.0)])
    #param[7] = 180-x
    #params.append([np.cos(np.pi*param[0]/180.0),np.sin(np.pi*param[0]/180.0),np.cos(np.pi*param[7]/180.0),np.sin(np.pi*param[7]/180.0)])
    
params = np.stack(params)

inp = Input(shape=(128,))

out = Dense(4,activation='linear')(inp)
model = Model(inputs=inp, outputs=out)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(codes, params, batch_size=128, epochs=1000)

model.save('encoder-regression.h5')

# test
result = model.predict(codes)
f = open('encoder/regression.txt', "w")
inputs = []
outputs = []
for id in range(dataset_size):
    inp1 = np.arctan2(params[id][0],params[id][1])*180.0/np.pi
    inp2 = np.arctan2(params[id][2],params[id][3])*180.0/np.pi
    out1 = np.arctan2(result[id][0],result[id][1])*180.0/np.pi
    out2 = np.arctan2(result[id][2],result[id][3])*180.0/np.pi
    f.write('{:04d},{:2.0f},{:2.0f},{:2.0f},{:2.0f}\n'.format(id,inp1,inp2,out1,out2))
    inputs.append([(inp1//5)*5,(inp2//5)*5])
    outputs.append([(out1//5)*5,(out2//5)*5])

f.close()

ok=[0,0]
almost=[0,0]
for id in range(dataset_size):
    for k in range(2):
        if inputs[id][k] == outputs[id][k]:
            ok[k] += 1
        elif np.abs(inputs[id][k]-outputs[id][k]) <= 5:
            almost[k] += 1
        
for k in range(2):
    print(k,': ',100.0*ok[k]/dataset_size,'% ',100.0*almost[k]/dataset_size,'% ',100.0*(ok[k]+almost[k])/dataset_size,'%')

