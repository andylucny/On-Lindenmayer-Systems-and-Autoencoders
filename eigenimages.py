import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset_path = "dataset/rose"
dataset_size = 1498
shape = (28, 28)
images = []
for id in range(dataset_size):
    img = cv2.imread(dataset_path+'{:04d}.png'.format(id),cv2.IMREAD_GRAYSCALE)
    images.append(img.flatten())
    #img = np.flip(img,axis=0)
    #images.append(img.flatten())
    #img = np.flip(img,axis=1)
    #images.append(img.flatten())
    #img = np.flip(img,axis=0)
    #images.append(img.flatten())

images = np.asarray(images,np.float)/255.0

print('please wait ...')
N = 8
mean, eigenvectors = cv2.PCACompute(images, mean=None, maxComponents=N)
print('... done')

average = mean.reshape(shape)
cv2.imwrite("eigenimages/average.png",np.asarray(average*255,np.uint8))

eigenimages = []
for v in range(N):
    eigenimage = eigenvectors[v].reshape(shape)
    cv2.imwrite("eigenimages/eigen"+str(v)+".png",np.asarray(eigenimage*255,np.uint8))
    eigenimages.append(eigenimage)

def dummyHandler(*args):
    pass
    
cv2.namedWindow("Eigenimages",cv2.WINDOW_AUTOSIZE)
average_toshow = cv2.resize(np.asarray(average*255,np.uint8),(112,112))
cv2.imshow("Eigenimages",cv2.hconcat([average_toshow,average_toshow]))
for v in range(N):
    cv2.createTrackbar("weight"+str(v),"Eigenimages",100,200,dummyHandler)

id=0
while True:

    weights = []
    image = np.copy(average)
    for v in range(N):
        weight = 5*(cv2.getTrackbarPos("weight"+str(v),"Eigenimages")-100)
        weights.append(weight)
        image = np.add(image,weight*eigenimages[v])
    
    image[image < 0] = 0
    image[image > 1] = 1
    #image[image > 0] = 1
    
    image_toshow = cv2.resize(np.asarray(image*255,np.uint8),(112,112))
    cv2.imshow("Eigenimages",cv2.hconcat([image_toshow,average_toshow]))
    key = cv2.waitKey(10)
    if key == 27:
        break

    elif key == ord('s'):
        cv2.imwrite('eigenimages/image'+str(id)+'.png',image_toshow)
        np.savetxt('eigenimages/image'+str(id)+'.txt',np.array(weights))

    elif key == ord('r'):
        for v in range(N):
            cv2.setTrackbarPos("weight"+str(v),"Eigenimages",100); 

cv2.destroyAllWindows()

print('please wait ...')
covariance_matrix, mean = cv2.calcCovarMatrix(images, None, cv2.COVAR_ROWS | cv2.COVAR_NORMAL | cv2.COVAR_SCALE)
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
plt.savefig('eigenimages\eigenvalues.png')

plt.figure(figsize=[8,6])
plt.plot(eigenvalues[:30],'r',linewidth=2.0)
plt.legend(['abs(eigenvalues)'],fontsize=18)
plt.xlabel('order ',fontsize=16)
plt.ylabel('abs',fontsize=16)
plt.title('Eigenvalues',fontsize=16)
plt.savefig('eigenimages\eigenvalues2.png')
