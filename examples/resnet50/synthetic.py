# code coming from Resnet-keras tutorial
# https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/resnets_utils.py

import numpy as np
import h5py
import sys
import random
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#sys.path.append('build/lib.linux-x86_64-3.6')
#sys.path.append('build/')
#os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import tensorflow as tf

sys.setrecursionlimit(1500)

def load_dataset():
    return  tf.data.Dataset.from_tensor_slices([1, 2, 3])





def create_keras_model(size_per_layer, num_layers, variance, num_transferred):
    random.seed(2)
    model_id=0
    assert size_per_layer - variance > 0
    inputs = tf.keras.Input(32, name="input")
    for i in range(num_layers+1):
        layer_size = random.randint(
                size_per_layer - variance, size_per_layer + variance
            )
        name = f'transfer{i}{model_id}{variance}{num_layers}'
        if i==0:
            x = tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal')(inputs)
        elif i==num_layers:
            outputs = tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal')(x)
        else:
            x= tf.keras.layers.Dense(layer_size, name=name, kernel_initializer='random_normal')(x)
    model = keras.Model(inputs, outputs)
    return model



def create_model(size_per_layer, num_layers, variance, num_transferred):
    model = create_keras_model(size_per_layer, num_layers, variance, num_transferred)

    return model


def build_model(model, optimizer='Adam'):
    """
    Compiles the model.
    optimizer can be 'Adam' or 'SGD'.
    """
    opt = None
    if optimizer == 'Adam':
        opt = Adam(lr=0.0001)
    elif optimizer == 'SGD':
        opt = SGD(lr=0.0001, momentum=0.9, nesterov=False)
    else:
        print('Unkown optimizer '+optimizer)
        sys.exit(-1)


def train_model(model, dataset, batch_size=32,
                epochs=1, callbacks=[], verbose=1):
    model.fit(dataset['x_train'], dataset['y_train'],
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              callbacks=callbacks)


def evaluate_model(model, dataset, verbose=0):
    score = model.evaluate(dataset['x_test'],
                           dataset['y_test'],
                           verbose=verbose)
