import numpy as np
from keras.datasets import fashion_mnist

def as_quantized_double_gray(X_rgb_int, n_colors):
    X_out = X_rgb_int.mean(axis=-1)//(256//n_colors)/n_colors
    X_out /= ((n_colors-1)/n_colors)
    X_out = np.expand_dims(X_out, axis=-1)
    
    return X_out

def gray2rgb(X_gray):    
    new_shape = (X_gray.shape[0], X_gray.shape[1], X_gray.shape[2], 3)
    X_rgb = np.zeros(new_shape)
    X_rgb[..., 0] = X_gray[..., 0]
    X_rgb[..., 1] = X_gray[..., 0]
    X_rgb[..., 2] = X_gray[..., 0]
    
    return X_rgb

def load_fashion_mnist_32x32():
    # load dataset using keras
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # pad images
    X_train = np.pad(X_train, [(0, 0), (2, 2), (2, 2)], 'constant', constant_values=0)
    X_test = np.pad(X_test, [(0, 0), (2, 2), (2, 2)], 'constant', constant_values=0)

    return (X_train, y_train), (X_test, y_test)

