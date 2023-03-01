import glob
import os
import random
from skimage import exposure
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# import keras library
import keras

import glob
# import Sequential from the keras models module
from keras.models import Sequential

# import Dense, Dropout, Flatten, Conv2D, MaxPooling2D from the keras layers module
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
D_PATH = '../CNN_pytorch/data/'

def tensorflow_model():
    # define model as Sequential
    model = Sequential()
    # first convolutional layer with 32 filters
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=[41, 41, 3]))
    # reduce dimensionality through max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add a second 2D convolutional layer with 64 filters
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # add dropout to prevent over fitting
    model.add(Dropout(0.25))
    # add a thirs 2D convolutional layer with 128 filters
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    # fully connected layer
    model.add(Dense(128, activation='relu'))
    # add additional dropout to prevent overfitting
    model.add(Dropout(0.5))
    # prediction layers
    model.add(Dense(1, activation='sigmoid', name='preds'))
    # show model summary
    model.summary()

    # Compile the model
    model.compile(
        # set the loss as binary_crossentropy
        loss='binary_crossentropy',
        # set the optimizer as stochastic gradient descent
        optimizer='adam',
        # set the metric as accuracy
        metrics=['accuracy']
    )
    model = keras.models.load_model(os.path.join( '../tensorflow_model/MI_CNN_model_0301.h5'))

    return model


if __name__=="__main__":
    model=tensorflow_model()
    # y_proba = model.predict(X_test)
    #
    # y_pred = np.round(y_proba).astype('int')
    # y_pred
