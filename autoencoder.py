import numpy as np
import cv2
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Activation
from keras.models import Model
from keras import backend as K
#from keras.utils import plot_model
import matplotlib.pyplot as plt
import random

# load custom dataset
dataset_path = "dataset/rose"
dataset_size = 1498
dataset = []
shuffled_dataset = []
for id in range(dataset_size):
    img = cv2.imread(dataset_path+'{:04d}.png'.format(id),cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.round(img)
    dataset.append(img.reshape(img.shape[0],img.shape[1],1))
    shuffled_dataset.append(img.reshape(img.shape[0],img.shape[1],1))
    #img = np.flip(img,axis=0)
    #shuffled_dataset.append(img.reshape(img.shape[0],img.shape[1],1))
    #img = np.flip(img,axis=1)
    #shuffled_dataset.append(img.reshape(img.shape[0],img.shape[1],1))
    #img = np.flip(img,axis=0)
    #shuffled_dataset.append(img.reshape(img.shape[0],img.shape[1],1))

random.shuffle(shuffled_dataset)
m=(len(shuffled_dataset)*9)//10
x_train = np.stack(shuffled_dataset[:m])
x_test = np.stack(shuffled_dataset[m:])

inp = Input(shape=(28, 28, 1))                                  # 28x28 x 1
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)  # 28x28 x 16
x = MaxPooling2D((2, 2), padding='same')(x)                     # 14x14 x 16
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)     # 14x14 x 8
x = MaxPooling2D((2, 2), padding='same')(x)                     # 7x7 x 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)     # 7x7 x 8
x = MaxPooling2D((2, 2), padding='same')(x)                     # 4x4 x 8
latent_space = Activation('sigmoid')(Flatten()(x))              # 128
x = Reshape((4,4,8))(latent_space)                              # 4x4 x 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)     # 4x4 x 8
x = UpSampling2D((2, 2))(x)                                     # 8x8 x 8
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)     # 8x8 x 8
x = UpSampling2D((2, 2))(x)                                     # 16x16 x 8
x = Conv2D(16, (3, 3), activation='relu')(x)                    # 14x14 x 16
x = UpSampling2D((2, 2))(x)                                     # 28x28 x 16
out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)# 28x28 x 1

autoencoder = Model(inp, out)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()
#plot_model(autoencoder, show_shapes=True, to_file='autoencoder.png')

history = autoencoder.fit(x_train, x_train,epochs=200,batch_size=128,shuffle=True,validation_data=(x_test, x_test))

plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=2.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('training.png')

scores = autoencoder.evaluate(x_test, x_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# save model and weights
autoencoder.save('roses_autoencoder.h5')

# use the model on the samples
input_images = np.stack(dataset)
output_images = autoencoder.predict(input_images)
path = 'autoencoder/rose'
for id in range(output_images.shape[0]):
    cv2.imwrite(path+'{:04d}.png'.format(id),np.asarray(output_images[id]*255,np.uint8))

# use the first half of autoencoder as an encoder
encoder = Model(inp, latent_space)
encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
encoder.summary()
encoder.save('roses_encoder.h5')

coded = encoder.predict(input_images)
print(coded[0])

# save codes for later PCA analysis
path = 'encoder/rose'
input_images = np.stack(dataset)
output_codes = encoder.predict(input_images)
for id in range(output_codes.shape[0]):
    np.savetxt(path+'{:04d}.txt'.format(id),output_codes[id])
#for id in range(input_images.shape[0]):
#    input_images[id] = np.flip(input_images[id],axis=0)
#output_codes = encoder.predict(input_images)
#for id in range(output_codes.shape[0]):
#    np.savetxt(path+'{:04d}v.txt'.format(id),output_codes[id])
#for id in range(input_images.shape[0]):
#    input_images[id] = np.flip(input_images[id],axis=1)
#output_codes = encoder.predict(input_images)
#for id in range(output_codes.shape[0]):
#    np.savetxt(path+'{:04d}vh.txt'.format(id),output_codes[id])
#for id in range(input_images.shape[0]):
#    input_images[id] = np.flip(input_images[id],axis=0)
#output_codes = encoder.predict(input_images)
#for id in range(output_codes.shape[0]):
#    np.savetxt(path+'{:04d}h.txt'.format(id),output_codes[id])

# use the second half of autoencoder as a decoder
for i in range(len(autoencoder.layers)):
    if autoencoder.layers[i].name == 'reshape_1':
        break
        
code_shape = autoencoder.layers[i].get_input_shape_at(0)
print(code_shape)

coded_input = Input(shape=(128,), dtype=float)
x = coded_input
for layer in autoencoder.layers[i:]:
    x = layer(x)

decoder = Model(coded_input, x)
decoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
decoder.summary()
decoder.save('roses_decoder.h5')

decoded = decoder.predict(coded)
print(decoded.shape)
path = 'decoder/rose'
for id in range(decoded.shape[0]):
    cv2.imwrite(path+'{:04d}.png'.format(id)+".png",np.asarray(decoded[id]*255,np.uint8))
