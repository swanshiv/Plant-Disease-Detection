# Intialization of Program. by Importing various LIbraries
import numpy as np
import matplotlib.pyplot as plt

# here we are working on Tensorflow version 2.1.0 so we need to write tensorflow.keras.
#keras is in built function in Tensorflow .
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image

# For checking out that how many images are available in the train set we can use import OS
for types in os.listdir("D:\pant\drive\My Drive/train_set/"):
    print(str(len(os.listdir("D:\pant\drive\My Drive/train_set/"+ types)))+" "+ types+' images')

# Complete Dataset images can be loaded using ImageDataGenerator function
img_size=48
batch_size=64
datagen_train=ImageDataGenerator(horizontal_flip=True)
train_generator=datagen_train.flow_from_directory("D:\pant\drive\My Drive/train_set",
target_size=(img_size,img_size),
batch_size=batch_size,
class_mode='categorical',
shuffle=True)

datagen_test=ImageDataGenerator(horizontal_flip=True)
validation_generator=datagen_test.flow_from_directory("D:\pant\drive\My Drive/test_data",
target_size=(img_size,img_size),
batch_size=batch_size,
class_mode='categorical',
shuffle=True)

detection=Sequential()

#convolutional layer-1
detection.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,3)))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#2 -convolutional layer-2
detection.add(Conv2D(128,(5,5),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#3 -convolutional layer-3
detection.add(Conv2D(256,(3,3),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#4 -convolutional layer-4
detection.add(Conv2D(512,(3,3),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

#5 -convolutional layer-5
detection.add(Conv2D(512,(3,3),padding='same'))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(MaxPooling2D(pool_size=(2,2)))
detection.add(Dropout(0.25))

detection.add(Flatten())
detection.add(Dense(256))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(Dropout(0.25))

detection.add(Dense(512))
detection.add(BatchNormalization())
detection.add(Activation('relu'))
detection.add(Dropout(0.25)

detection.add(Dense(15,activation='softmax'))
optimum=Adam(lr=0.005)
#lr-learning rate
detection.compile(optimizer=optimum,loss='categorical_crossentropy',metrics=['accuracy'])

detection.summary()
# This is used to get all the summary related to model , added layers descriptions.

ephocs=10
steps_per_epoch=train_generator.n//train_generator.batch_size
steps_per_epoch
validation_steps=validation_generator.n//validation_generator.batch_size
validation_steps
detection.fit(x=train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=ephocs,
                    validation_data=validation_generator,
                    validation_steps=validation_steps)
detection.save('Plant_Disease_Detection.h5')

from tensorflow.keras.models import load_model
Detection=load_model('Plant_Disease_Detection.h5')
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

test_img=image.load_img("D:\pant\drive\My Drive/test_data/Tomato_healthy/0d515778-61ef-4f0b-ab54-75607c80220f___RS_HL 9745.jpg",target_size=(48,48))

plt.imshow(test_img)
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
result=Detection.predict(test_img)
a=result.argmax()

# print('a:',a)
classes=train_generator.class_indices

# print(classes)
# print(len(classes))

category=[]
for i in classes:
          category.append(i)
for i in range(len(classes)):
          if(i==a):
              output=category[i]
output

from tensorflow.keras.models import load_model
Detection=load_model('Plant_Disease_Detection.h5')
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
test_img=image.load_img("D:\pant\drive\MyDrive/test_data/Pepper__bell___healthy/1dd1b153-8ded-439f-8c9e-c9970c67e642___JR_HL 8163.jpg",target_size=(48,48))
plt.imshow(test_img)
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
result=Detection.predict(test_img)
a=result.argmax()
# print('a',a)
classes=train_generator.class_indices
category=[]
for i in classes:
          category.append(i)
for i in range(len(classes)):
           if(i==a):
                output=category[i]
output  
