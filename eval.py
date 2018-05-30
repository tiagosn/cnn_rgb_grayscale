import os

import numpy as np
import pandas as pd

from glob import glob
from keras.datasets import cifar10, cifar100
from keras.models import Model, load_model
from sklearn.linear_model import SGDClassifier

from simple_cnn import *
from utils import *

# Note on CIFAR-100 labels:
#   - coarse: 20 classes
#   - fine: 100 classes

def eval_cifar10(model_path, ds_name, df_results, rgb_model=True, noise_level=None):
    n_colors = [256, 128, 64, 32, 16, 8]
    if noise_level is not None:
        n_colors = [256]

    model = load_model(model_path)
    
    model_features = load_model(model_path)
    if 'simple' in model_path:
        model_features.pop()
        model_features.pop()
        model_features.pop()
    elif 'all_conv' in model_path:
        model_features.pop()
        model_features.pop()
    elif 'resnet20' in model_path:
        model_features = Model(model_features.input, model_features.layers[-2].output)

    # load dataset
    X_train, X_test = None, None
    y_train_svm, y_test_svm = None, None
    if ds_name == 'cifar10':
        (X_train, y_train_svm), (X_test, y_test_svm) = cifar10.load_data()
    elif ds_name == 'cifar100_fine':
        (X_train, y_train_svm), (X_test, y_test_svm) = cifar100.load_data(label_mode='fine')
    elif ds_name == 'cifar100_coarse':
        (X_train, y_train_svm), (X_test, y_test_svm) = cifar100.load_data(label_mode='coarse')
    elif ds_name == 'fashion_mnist':
        if noise_level is None:
            (X_train, y_train_svm), (X_test, y_test_svm) = load_fashion_mnist_32x32()
        else:
            (X_train, y_train_svm), (X_test, y_test_svm) = load_fashion_mnist_32x32(noise_level)
            ds_name = '%s-%d' % (ds_name, noise_level)

    # convert class vectors to binary class matrices (only for cifar10)
    y_train, y_test = None, None
    if ds_name == 'cifar10':
        y_train = keras.utils.to_categorical(y_train_svm, 10)
        y_test = keras.utils.to_categorical(y_test_svm, 10)

    if rgb_model:
        X_aux_train = X_train.copy()
        X_aux_train = X_aux_train.astype('float32')
        X_aux_train /= 255
        X_aux_test = X_test.copy()
        X_aux_test = X_aux_test.astype('float32')
        X_aux_test /= 255

        if ds_name == 'cifar10':
            loss, acc = model.evaluate(X_aux_test, y_test, verbose=1)
            df_results.loc[len(df_results)] = [model_path.split('/')[-1], ds_name, 'cnn', 'rgb', 256, loss, acc]      
            print('model: %s, dataset: %s, colors: %s, acc: %.2lf' % ('CNN_'+model_path.split('/')[-1], ds_name, 'rgb', acc))

        X_f_train = model_features.predict(X_aux_train, batch_size=32)
        X_f_test = model_features.predict(X_aux_test, batch_size=32)

        svm_sgd = SGDClassifier(loss='log', max_iter=100, n_jobs=-1)
        svm_sgd.fit(X_f_train, y_train_svm.reshape(-1))
        acc = svm_sgd.score(X_f_test, y_test_svm.reshape(-1))
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], ds_name, 'lr', 'rgb', 256, '-', acc]
        print('model: %s, dataset: %s, colors: %s, acc: %.2lf' % ('LR_'+model_path.split('/')[-1], ds_name, 'rgb', acc))

    for nc in n_colors:        
        X_aux_train = X_train.copy()
        X_aux_train = X_aux_train.astype('float32')
        X_aux_train = as_quantized_double_gray(X_aux_train, nc)
        X_aux_test = X_test.copy()
        X_aux_test = X_aux_test.astype('float32')
        X_aux_test = as_quantized_double_gray(X_aux_test, nc)
        if rgb_model:
            X_aux_train = gray2rgb(X_aux_train)
            X_aux_test = gray2rgb(X_aux_test)

        if ds_name == 'cifar10':
            loss, acc = model.evaluate(X_aux_test, y_test, verbose=1)
            df_results.loc[len(df_results)] = [model_path.split('/')[-1], ds_name, 'cnn', 'gray', nc, loss, acc]
            print('model: %s, dataset: %s, colors: %s, acc: %.2lf' % ('CNN_'+model_path.split('/')[-1], ds_name, 'gray'+str(nc), acc))

        X_f_train = model_features.predict(X_aux_train, batch_size=32)
        X_f_test = model_features.predict(X_aux_test, batch_size=32)

        svm_sgd = SGDClassifier(loss='log', max_iter=100, n_jobs=-1)
        svm_sgd.fit(X_f_train, y_train_svm.reshape(-1))
        acc = svm_sgd.score(X_f_test, y_test_svm.reshape(-1))
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], ds_name, 'lr', 'gray', nc, '-', acc]
        print('model: %s, dataset: %s, colors: %s, acc: %.2lf' % ('LR_'+model_path.split('/')[-1], ds_name, 'gray'+str(nc), acc))

    return df_results

ds_names = ['fashion_mnist', 'cifar100_coarse', 'cifar100_fine', 'cifar10']
model_paths = sorted(glob('saved_models/*.h5'))
for ds in ds_names:
    df_results = pd.DataFrame(columns=['model', 'dataset', 'clf', 'color_space', 'n_colors', 'test_loss', 'test_acc'])
    for mp in model_paths:
        if 'rgb' in mp:
            df_results = eval_cifar10(mp, ds, df_results, rgb_model=True)
        else:
            df_results = eval_cifar10(mp, ds, df_results, rgb_model=False)

    df_results.to_csv('results_%s.csv' % (ds), index=False)

df_results = pd.DataFrame(columns=['model', 'dataset', 'clf', 'color_space', 'n_colors', 'test_loss', 'test_acc'])
noise_levels = [10, 20, 30]
for nl in noise_levels:
    for mp in model_paths:
        if 'rgb' in mp:
            df_results = eval_cifar10(mp, 'fashion_mnist', df_results, rgb_model=True, noise_level=nl)
        else:
            df_results = eval_cifar10(mp, 'fashion_mnist', df_results, rgb_model=False, noise_level=nl)

df_results.to_csv('results_fashion_mnist-noise.csv', index=False)