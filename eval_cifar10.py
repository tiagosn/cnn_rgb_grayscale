import os

import numpy as np
import pandas as pd

from glob import glob
from keras.datasets import cifar10
from keras.models import load_model
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from simple_cnn import *
from utils import *

def eval_cifar10(model_path, df_results, rgb=True):
    num_classes = 10
    n_colors = [256, 128, 64, 32, 16, 8]

    model = load_model(model_path)
    
    model_features = load_model(model_path)
    model_features.pop()
    model_features.pop()
    model_features.pop()

    # load cifar-10
    (X_train, y_train_svm), (X_test, y_test_svm) = cifar10.load_data()

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train_svm, num_classes)
    y_test = keras.utils.to_categorical(y_test_svm, num_classes)

    if rgb:
        X_aux_train = X_train.copy()
        X_aux_train = X_aux_train.astype('float32')
        X_aux_train /= 255
        X_aux_test = X_test.copy()
        X_aux_test = X_aux_test.astype('float32')
        X_aux_test /= 255

        loss, acc = model.evaluate(X_aux_test, y_test, verbose=1)
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], 'cnn', 'rgb', 256, loss, acc]      
        print('[RGB] model: %s, acc: %.2lf' % ('CNN_'+model_path.split('/')[-1], acc))

        X_f_train = model_features.predict(X_aux_train, batch_size=32)
        X_f_test = model_features.predict(X_aux_test, batch_size=32)

        # lr = LogisticRegression(solver='saga', multi_class='multinomial', n_jobs=-1, random_state=42)
        # lr.fit(X_f_train, y_train_svm.reshape(-1))
        # acc = lr.score(X_f_test, y_test_svm.reshape(-1))
        svm_sgd = SGDClassifier(n_jobs=-1)
        svm_sgd.fit(X_f_train, y_train_svm.reshape(-1))
        acc = svm_sgd.score(X_f_test, y_test_svm.reshape(-1))
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], 'svm', 'rgb', 256, '-', acc]
        print('[RGB] model: %s, acc: %.2lf' % ('SVM_'+model_path.split('/')[-1], acc))

    for nc in n_colors:
        X_aux_train = X_train.copy()
        X_aux_train = X_aux_train.astype('float32')
        X_aux_train = as_quantized_double_gray(X_aux_train, nc)
        X_aux_test = X_test.copy()
        X_aux_test = X_aux_test.astype('float32')
        X_aux_test = as_quantized_double_gray(X_aux_test, nc)
        if rgb:
            X_aux_train = gray2rgb(X_aux_train)
            X_aux_test = gray2rgb(X_aux_test)

        loss, acc = model.evaluate(X_aux_test, y_test, verbose=1)
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], 'cnn', 'gray', nc, loss, acc]
        print('[GRAY %d] model: %s, acc: %.2lf' % (nc, model_path.split('/')[-1], acc))

        X_f_train = model_features.predict(X_aux_train, batch_size=32)
        X_f_test = model_features.predict(X_aux_test, batch_size=32)

        svm_sgd = SGDClassifier(n_jobs=-1)
        svm_sgd.fit(X_f_train, y_train_svm.reshape(-1))
        acc = svm_sgd.score(X_f_test, y_test_svm.reshape(-1))
        df_results.loc[len(df_results)] = [model_path.split('/')[-1], 'svm', 'gray', nc, '-', acc]
        print('[GRAY %d] model: %s, acc: %.2lf' % (nc, 'SVM_'+model_path.split('/')[-1], acc))

    return df_results

df_results = pd.DataFrame(columns=['model', 'clf', 'color_space', 'n_colors', 'test_loss', 'test_acc'])

model_paths = sorted(glob('saved_models/*.h5'))
for mp in model_paths:
    if 'rgb' in mp:
        df_results = eval_cifar10(mp, df_results, rgb=True)
    else:
        # nc = int(mp.split('/')[-1].replace('simple_cifar10_gray', '').replace('.h5', ''))
        df_results = eval_cifar10(mp, df_results, rgb=False)

df_results.to_csv('results_cifar10.csv')
