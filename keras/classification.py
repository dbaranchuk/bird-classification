import numpy as np
from skimage.transform import resize, rotate

from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from keras.utils.np_utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras_vgg import vgg16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import Callback

from numpy.random import uniform as rnd

IMAGE_SIZE = 224

NUM_CLASSES = 50
PER_CLASS = 40
TRAIN_SIZE_PER_CLASS = 40 #30

BATCH_SIZE = 32
N_EPOCH = 25

model_dir = '/Users/dmitrybaranchuk/cv/cvintro2016/hw-07/vgg16_weights.h5'

P = 0.5

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = (optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations))).eval()
        print('\nLR: {:.8f}'.format(float(lr)))

        
class BirdClassifier:
    def __init__(self):
        self.model = vgg16(model_dir)
        self.model.pop()
        self.model.pop()
        self.model.add(Dense(NUM_CLASSES, name='new_dense'))
        self.model.add(Activation('softmax', name='softmax'))

    def set_data(self, X, y=None):
        if y is None:
            self.X_test = X
        else:
            self.X_train, self.y_train = (X, y) 
            # Split data to train and val  
            #self.X_train, self.y_train = ([], [])
            #self.X_val, self.y_val = ([], [])
            #            X, y = list(X), list(y)
        
            #for i in range(NUM_CLASSES):
            #       self.X_train += X[i*PER_CLASS: i*PER_CLASS + TRAIN_SIZE_PER_CLASS]
            #       self.y_train += y[i*PER_CLASS: i*PER_CLASS + TRAIN_SIZE_PER_CLASS]
    
            #       self.X_val += X[i*PER_CLASS + TRAIN_SIZE_PER_CLASS: (i+1)*PER_CLASS]
            #           self.y_val += y[i*PER_CLASS + TRAIN_SIZE_PER_CLASS: (i+1)*PER_CLASS]
    
            #self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
            #self.X_val, self.y_val = np.array(self.X_val), np.array(self.y_val)
           

    def augment_data(self):
        X_aug, y_aug = ([],[])
        
        for i in range(len(self.X_train)):
            new_X = [self.X_train[i]]
            new_y = [self.y_train[i]]
            
            if rnd(0, 1) < P:
                new_X.append(new_X[0][:,::-1,:])
                new_y.append(new_y[0])
                    
            X_aug += new_X
            y_aug += new_y
                
        self.X_train = np.array(X_aug)
        self.y_train = np.array(y_aug)

    def train(self, mode):
        if mode == 'FC':
            for layer in self.model.layers[:24]:
                layer.trainable = False
            
            sgd = SGD(lr=0.0001, decay=1e-3, momentum=0.9)
            self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE,
                           callbacks=[SGDLearningRateTracker()],
                           nb_epoch=N_EPOCH)#, validation_data=(self.X_val, self.y_val))

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
    X = preprocess_input(255.*X)
    y = to_categorical(y-1, NUM_CLASSES)
    print("Preprocessing is completed")
    
    model.set_data(X, y)
    model.augment_data()
    print("Augmentation is completed")
    
    model.train('FC')
    
    return model

def predict(model, X, b):
    X = resize_images(X, b)
    X = preprocess_input(255.*X)
    
    model.set_data(X)
    return model.predict()+1
