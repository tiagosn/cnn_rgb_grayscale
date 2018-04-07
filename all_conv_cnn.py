# https://github.com/tiagosn/dnnnoise2017/blob/master/networks.py

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

def create_all_conv_net_ref_c(in_shape, n_classes):
    model = Sequential([
        Conv2D(96, (3, 3), padding='valid', input_shape=in_shape),
        Activation("relu"),
        Conv2D(96, (3, 3), padding='valid'),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
        Activation("relu"),
        Dropout(0.5),

        Conv2D(96, (3, 3), padding='valid'),
        Activation("relu"),
        Conv2D(96, (3, 3), padding='valid'),
        MaxPooling2D(pool_size=(3, 3), strides=(2,2)),
        Activation("relu"),
        Dropout(0.5),

        Conv2D(192, (3, 3), padding='valid'),
        Activation("relu"),
        Conv2D(192, (1, 1), padding='valid'),
        Activation("relu"),
        Conv2D(n_classes, (1, 1), padding='valid'),
        Activation("relu"),
        Flatten(),

        Dense(10),
        Activation('softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model