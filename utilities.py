import pickle
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def print_predictions(actual, predicted, name_file):
    
    strpr = "actual;predicted\n"

    for ix in range(len(actual)):

        for ix2 in range(len(actual[ix])):

            strpr += str(actual[ix][ix2]) + ";" + str(predicted[ix][ix2]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", ""))
    file_processed.close()

def get_XY(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    Y = [] 
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        if len(x_vals) == time_steps and i + time_steps + len_output <= len(dat):
            X.append(np.array(x_vals))
            Y.append(np.array(dat[i + time_steps:i + time_steps + len_output]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def get_X(dat, time_steps, len_skip = -1, len_output = -1):
    X = []
    if len_skip == -1:
        len_skip = time_steps
    if len_output == -1:
        len_output = time_steps
    for i in range(0, len(dat), len_skip):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        if len(x_vals) == time_steps:
            X.append(np.array(x_vals))
    X = np.array(X)
    return X
  
def custom_activation_abs(x):
    return K.abs(x)

def custom_activation_waves(x):
    return K.relu(x, max_value = 0.6, threshold = -0.6)  

def custom_activation_amplitude(x):
    return K.relu(K.abs(x), max_value = 0.1) + 0.05

def custom_activation_middle(x):
    return K.relu(K.abs(x), max_value = 0.1)

def add_custom():
    get_custom_objects().update({'custom_activation_abs': Activation(custom_activation_abs)})
    get_custom_objects().update({'custom_activation_waves': Activation(custom_activation_waves)})
    get_custom_objects().update({'custom_activation_amplitude': Activation(custom_activation_amplitude)})
    get_custom_objects().update({'custom_activation_middle': Activation(custom_activation_middle)})

def create_RNN(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    add_custom()
    model.add(SimpleRNN(hidden_units, input_shape = input_shape, activation = "linear"))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def create_GRU(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    add_custom()
    model.add(GRU(hidden_units, input_shape = input_shape, activation = "linear"))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def create_LSTM(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    add_custom()
    model.add(LSTM(hidden_units, input_shape = input_shape, activation = "linear"))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data