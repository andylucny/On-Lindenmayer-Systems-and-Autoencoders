import numpy as np
import cv2
from keras.layers import Input
from keras.models import Model
from keras.models import load_model

model_name = 'roses_decoder.h5'
decoder = load_model(model_name)
decoder.summary()

path = 'encoder/rose'
id=25 # sample code
features0 = np.loadtxt(path+'{:04d}.txt'.format(id))
id=26 # sample code
features1 = np.loadtxt(path+'{:04d}.txt'.format(id))
id=2 # sample code
features2 = np.loadtxt(path+'{:04d}.txt'.format(id))
id=235 # sample code
features3 = np.loadtxt(path+'{:04d}.txt'.format(id))
features = np.copy(features0)

last_features = np.copy(features)
last_features[0] = 1.0

cv2.namedWindow("generator")
cv2.imshow('generator',np.zeros((112*5,112*5),np.uint8))

index = np.argmax(np.abs(features0-features1))
feature = int(127*features[index])

def update_index( *args ):
    global index, feature
    index = args[0]
    cv2.setTrackbarPos("value", "generator", int(127*features[index]))

def update_feature( *args ):
    global index, features
    print(index,args[0])
    feature = float(args[0]) / 127.0
    features[index] = feature

cv2.createTrackbar("index", "generator", index, 127, update_index)
cv2.createTrackbar("value", "generator", feature, 127, update_feature)

id = 0
while True:

    if not np.array_equal(features,last_features):
        last_features = np.copy(features)
        coded = np.array([features],np.float)
        decoded = decoder.predict(coded)
        decoded = np.asarray(decoded[0]*255,np.uint8)
        decoded = cv2.resize(decoded,(112*5,112*5))
        #print('displayed')
        cv2.imshow('generator',decoded)
        
    key = cv2.waitKey(10)
    if key == 27:
        break
    elif key == ord('r') or key == ord('0'):
        features = features = np.copy(features0)
        update_index(index)
    elif key == ord('1'):
        features = features = np.copy(features1)
        update_index(index)
    elif key == ord('2'):
        features = features = np.copy(features2)
        update_index(index)
    elif key == ord('3'):
        features = features = np.copy(features3)
        update_index(index)
    elif key == ord('s'):
        cv2.imwrite('generator/generated'+str(id)+'.png',decoded)
        np.savetxt('generator/generated'+str(id)+'.txt',features)
        id += 1

cv2.destroyAllWindows()
