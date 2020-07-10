import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model_name = 'roses_decoder.h5'
decoder = load_model(model_name)
decoder.summary()

dataset_size = 1498  
codes = []
path = 'encoder/rose'
for id in range(dataset_size):
    code = np.loadtxt(path+'{:04d}.txt'.format(id))
    codes.append(code)
#    code = np.loadtxt(path+'{:04d}v.txt'.format(id))
#    codes.append(code)
#    code = np.loadtxt(path+'{:04d}vh.txt'.format(id))
#    codes.append(code)
#    code = np.loadtxt(path+'{:04d}h.txt'.format(id))
#    codes.append(code)
    
codes = np.stack(codes)

print('please wait ...')
N = 8
mean, eigenvectors = cv2.PCACompute(codes, mean=None, maxComponents=N)
print('... done')

shape = (28,28)
average = mean.reshape((mean.shape[1],))
average_image = decoder.predict(np.array([average],np.float))[0].reshape(shape)
cv2.imwrite("eigencodes/average.png",np.asarray(average_image*255,np.uint8))

eigenimages = []
for v in range(N):
    eigenimage = decoder.predict(np.array([eigenvectors[v]],np.float))[0].reshape(shape)
    cv2.imwrite("eigencodes/eigen"+str(v)+".png",np.asarray(eigenimage*255,np.uint8))
    eigenimages.append(eigenimage)
    
eigenimages = np.stack(eigenimages)

def dummyHandler(*args):
    pass
    
cv2.namedWindow("Eigenimages",cv2.WINDOW_AUTOSIZE)
average_toshow = cv2.resize(np.asarray(average_image*255,np.uint8),(112,112))
cv2.imshow("Eigenimages",cv2.hconcat([average_toshow,average_toshow]))
for v in range(N):
    cv2.createTrackbar("weight"+str(v),"Eigenimages",100,200,dummyHandler)

id = 0
while True:

    weights = []
    features = np.copy(average)
    for v in range(N):
        weight = 0.1*(cv2.getTrackbarPos("weight"+str(v),"Eigenimages")-100)
        weights.append(weight)
        features = np.add(features,weight*eigenvectors[v])
    
    image = decoder.predict(np.array([features],np.float))[0].reshape(shape)
    
    image[image < 0] = 0
    image[image > 1] = 1
    #image[image > 0] = 1
    
    image_toshow = cv2.resize(np.asarray(image*255,np.uint8),(112,112))
    cv2.imshow("Eigenimages",cv2.hconcat([image_toshow,average_toshow]))
    key = cv2.waitKey(10)
    if key == 27:
        break

    elif key == ord('s'):
        cv2.imwrite('eigencodes/image'+str(id)+'.png',np.asarray(image*255,np.uint8))
        np.savetxt('eigencodes/image'+str(id)+'.txt',np.array(weights))
        id += 1

    elif key == ord('r'):
        for v in range(N):
            cv2.setTrackbarPos("weight"+str(v),"Eigenimages",100); 

cv2.destroyAllWindows()

print('please wait ...')
covariance_matrix, mean = cv2.calcCovarMatrix(codes, None, cv2.COVAR_ROWS | cv2.COVAR_NORMAL | cv2.COVAR_SCALE)
_, eigenvalues, eigenvectors = cv2.eigen(covariance_matrix)
print('... done')
eigenvalues = np.reshape(eigenvalues,(eigenvalues.shape[0],))
eigenvalues = np.abs(eigenvalues)
eigenvalues = np.flip(np.sort(eigenvalues))

plt.figure(figsize=[8,6])
plt.plot(eigenvalues,'r',linewidth=2.0)
plt.legend(['abs(eigenvalues)'],fontsize=18)
plt.xlabel('order ',fontsize=16)
plt.ylabel('abs',fontsize=16)
plt.title('Eigenvalues',fontsize=16)
plt.savefig('eigencodes/eigenvalues.png')

plt.figure(figsize=[8,6])
plt.plot(eigenvalues[:30],'r',linewidth=2.0)
plt.legend(['abs(eigenvalues)'],fontsize=18)
plt.xlabel('order ',fontsize=16)
plt.ylabel('abs',fontsize=16)
plt.title('Eigenvalues',fontsize=16)
plt.savefig('eigencodes/eigenvalues2.png')
