import pickle
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU

def print_predictions(actual, predicted, name_file):
    
    strpr = "actual;predicted\n"

    for ix in range(len(actual)):

        strpr += str(actual[ix]) + ";" + str(predicted[ix]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", ""))
    file_processed.close()

def get_XY(dat, time_steps, num_props):
    X = []
    Y = [] 
    for i in range(len(dat)):
        x_vals = dat[i:min(i + time_steps, len(dat))]
        if len(x_vals) == time_steps and i + time_steps < len(dat):
            X.append(np.array(x_vals))
            Y.append(np.array(dat[i + time_steps]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape = input_shape, activation = act_layer))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def create_GRU(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    model.add(GRU(hidden_units, input_shape = input_shape, activation = act_layer))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model

def create_LSTM(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape = input_shape, activation = act_layer))
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