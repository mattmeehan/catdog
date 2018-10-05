
from __future__ import division, print_function
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import (Dense, Dropout, Activation, 
        Flatten, Conv2D, MaxPooling2D, LeakyReLU)


def load_cnn(weights='/Users/matt/projects/catdog/cnn_deep_trained.h5', size=50):
    model = models.build_cnn(size)
    model.load_weights(weights)
    return model


def build_feed_forward_nn(img_size):
    """Build a simple feed-forward neural network.

    Parameters
    ----------
    img_size: int
        Length of the image in pixels (images are assumed square)

    Returns
    -------
    model : Sequential
        Compiled sequential model
    """

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(img_size, img_size)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


def build_cnn(img_size):
    """Build convolutional neural network.

    Note that CNN expects data with shape (img_size, img_size, n_chanels), 
    where n_channels=1 since we have converted to grayscale.

    Parameters
    ----------
    img_size: int
        Length of the image in pixels (images are assumed square)

    Returns
    -------
    model : Sequential
        Compiled CNN model
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_size,img_size,1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_size,img_size,1)))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model
