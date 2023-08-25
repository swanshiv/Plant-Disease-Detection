from tensorflow.keras import *

from IPython.display import SVG, Image

import os

from google.colab import drive
drive.mount("/content/gdrive")

import plant_disease_dataset_viewer

# Mention number of classes you want to work with
no_of_species = 2

def train_leaf_doctor(no_of_species, train_dataset_path, validation_dataset_path):
    
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
    detection.add(Dropout(0.25))

    detection.add(Dense(no_of_species,activation='softmax'))
    optimum=Adam(lr=0.005)
    #lr-learning rate
    detection.compile(optimizer=optimum,loss='categorical_crossentropy',metrics=['accuracy'])

    detection.summary()

    # Complete Dataset images can be loaded using ImageDataGenerator function
    img_size=48
    batch_size=32

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
    
    