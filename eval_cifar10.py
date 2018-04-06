import os

import numpy as np
import pandas as pd

from glob import glob
from keras.datasets import cifar10
from keras.models import load_model

from simple_cnn import *
from utils import *

def eval_cifar10(model_path, df_results, rgb=True):
    num_classes = 10
    n_colors = [256, 128, 64, 32, 16, 8]

    model = load_model(model_path)

    # load cifar-10
    (_, _), (X_test, y_test) = cifar10.load_data()

    # convert class vectors to binary class matrices.
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if rgb:
        X_aux = X_test.copy()
        X_aux = X_aux.astype('float32')
        X_aux /= 255

        loss, acc = model.evaluate(X_aux, y_test, verbose=1)
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], 'rgb', 256, loss, acc]      
        print('[RGB] model: %s, acc: %.2lf' % (model_path.split('/')[-1], acc))

    for nc in n_colors:
        X_aux = X_test.copy()
        X_aux = X_aux.astype('float32')
        X_aux = as_quantized_double_gray(X_aux, nc)
        if rgb:
            X_aux = gray2rgb(X_aux)

        loss, acc = model.evaluate(X_aux, y_test, verbose=1)
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], 'gray', nc, loss, acc]
        print('[GRAY %d] model: %s, acc: %.2lf' % (nc, model_path.split('/')[-1], acc))

    del model
    return df_results

df_results = pd.DataFrame(columns=['model', 'color_space', 'n_colors', 'test_loss', 'test_acc'])

model_paths = sorted(glob('saved_models/*.h5'))
for mp in model_paths:
    if 'rgb' in mp:
        df_results = eval_cifar10(mp, df_results, rgb=True)
    else:
        # nc = int(mp.split('/')[-1].replace('simple_cifar10_gray', '').replace('.h5', ''))
        df_results = eval_cifar10(mp, df_results, rgb=False)

df_results.to_csv('results_cifar10.csv', index=False)
