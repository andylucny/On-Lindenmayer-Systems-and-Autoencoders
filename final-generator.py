import numpy as np
import cv2
from keras.layers import Input
from keras.models import Model
from keras.models import load_model

decoder = load_model('roses_decoder.h5')
perceptron = load_model('decoder-perceptron.h5')
  
path = 'dataset/rose'
id=25 # sample code
param0 = np.loadtxt(path+'{:04d}.txt'.format(id))
id=26 # sample code
param1 = np.loadtxt(path+'{:04d}.txt'.format(id))
id=2 # sample code
param2 = np.loadtxt(path+'{:04d}.txt'.format(id))
id=235 # sample code
param3 = np.loadtxt(path+'{:04d}.txt'.format(id))
param = np.copy(param0)

last_value = -1

cv2.namedWindow("generator")
cv2.imshow('generator',np.zeros((112*5,112*5),np.uint8))

value = int(param[7])

def update_value( *args ):
    global value
    print(args[0])
    value = float(args[0])

cv2.createTrackbar("value", "generator", value, 90, update_value)

id = 0
while True:

    if last_value != value:
        last_value = value
        param[7] = value
        coded = perceptron.predict(param.reshape(1,-1))
        decoded = decoder.predict(coded)
        decoded = np.asarray(decoded[0]*255,np.uint8)
        decoded = cv2.resize(decoded,(112*5,112*5))
        cv2.imshow('generator',decoded)
        
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('r') or key == ord('0'):
        param = np.copy(param0)
        update_value(int(param[7]))
    elif key == ord('1'):
        param = np.copy(param1)
        update_value(int(param[7]))
    elif key == ord('2'):
        param = np.copy(param2)
        update_value(int(param[7]))
    elif key == ord('3'):
        param = np.copy(param3)
        update_value(int(param[7]))
    elif key == ord('s'):
        cv2.imwrite('generator/final'+str(id)+'.png',decoded)
        np.savetxt('generator/final'+str(id)+'.txt',param)
        id += 1

cv2.destroyAllWindows()
