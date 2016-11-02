import numpy as np
from skimage.transform import resize

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
import h5py

from keras_vgg import vgg16
from keras.applications.vgg16 import preprocess_input

IMAGE_SIZE = 224
VAL_SIZE = 500

NUM_CLASSES = 50
PER_CLASS = 30
TRAIN_SIZE_PER_CLASS = 20

BATCH_SIZE = 32
N_EPOCH = 10

model_dir = '/Users/dmitrybaranchuk/cv/cvintro2016/hw-07/vgg16_weights.h5'

class BirdClassifier:
    
    def __init__(self):
        self.model = vgg16(model_dir)
        self.model.pop()
        self.model.pop()
        self.model.add(Dense(NUM_CLASSES, name='new_dense'))
        self.model.add(Activation('softmax', name='softmax'))
        
        for i, layer in enumerate(self.model.layers):
            print(i, layer.name)

    def set_data(self, X, y=None):
        if y is None:
            self.X_test = X
        else:
            self.X_train, self.y_train = ([], [])
            self.X_val, self.y_val = ([], [])
            X, y = list(X), list(y)
            
            for i in range(NUM_CLASSES):
                self.X_train += X[i*PER_CLASS: i*PER_CLASS + TRAIN_SIZE_PER_CLASS]
                self.y_train += y[i*PER_CLASS: i*PER_CLASS + TRAIN_SIZE_PER_CLASS]
    
                self.X_val += X[i*PER_CLASS + TRAIN_SIZE_PER_CLASS: (i+1)*PER_CLASS]
                self.y_val += y[i*PER_CLASS + TRAIN_SIZE_PER_CLASS: (i+1)*PER_CLASS]
    
            self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
            self.X_val, self.y_val = np.array(self.X_val), np.array(self.y_val)
    
    def train(self, mode):
        if mode == 'FC':
            for layer in self.model.layers[:32]:
                layer.trainable = False
            
            sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
            self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE,
                       nb_epoch=N_EPOCH, validation_data=(self.X_val, self.y_val))

    def predict(self):
        return self.model.predict_classes(self.X_test, batch_size=BATCH_SIZE)


def resize_images(X, b):
    for i in range(len(X)):
        x, y, x_len, y_len = b[i].astype(int)
        
        X[i] = X[i][y:y+y_len, x:x+x_len]
        X[i] = resize(X[i], (IMAGE_SIZE, IMAGE_SIZE, 3))
    
    return np.array(X).transpose(0, 3, 1, 2)


def train_classifier(X, b, y):
    model = BirdClassifier()
    print("Initialization is completed")

    X = resize_images(X, b)
    X = preprocess_input(X)
    y = to_categorical(y-1, 50)
    print("Preprocessing is completed")
    
    print(X.shape, y.shape)
    
    model.set_data(X, y)
    model.train('FC')
    return model


def predict(model, X, b):
    X = resize_images(X, b)
    X = preprocess_input(X)
    
    model.set_data(X)
    return model.predict()+1
