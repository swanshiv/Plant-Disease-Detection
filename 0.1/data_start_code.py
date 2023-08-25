from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image

def step_data(img_size, batch_size, train_path, test_path):
    datagen_train=ImageDataGenerator(horizontal_flip=True)
    train_generator=datagen_train.flow_from_directory(train_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

    datagen_test=ImageDataGenerator(horizontal_flip=True)
    validation_generator=datagen_test.flow_from_directory(test_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

    return train_generator, validation_generator