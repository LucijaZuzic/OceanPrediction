import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utilities import get_XY, get_X, create_GRU, create_LSTM, create_RNN, print_predictions

ws = 365
models = ["LSTM", "RNN", "GRU"] 
hidden = 200
num_props = 1
ws_amplitude = 1
ws_middle = 1
ws_wave = 1

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    waves = list(file_data["sla"])  

    datetimes = list(file_data["date"])
        
    datetimes_value = [datetime.strptime(val, "%d.%m.%Y.") for val in datetimes]

    last_datetime = datetime(year = 2023, month = 8, day = 31)

    new_delta = last_datetime - datetimes_value[-1]

    for new_data in range(new_delta.days):
        datetimes_value.append(datetimes_value[-1] + timedelta(days = 1))

    datetimes_str = [datetime.strftime(val, "%d.%m.%Y.") for val in datetimes_value]
     
    datetimes_ix_filter = [i for i in range(0, len(datetimes_str), int(len(datetimes_str) // 6))]
    datetimes_filter = [datetimes_str[i] for i in datetimes_ix_filter]

    waves_train = list(file_data["sla"][:int(0.7 * len(waves) // ws * ws)]) 

    waves_test = list(file_data["sla"][int(0.7 * len(waves) // ws * ws):])

    middle_window_train = [np.average(waves_train[i:i+ws]) for i in range(0, len(waves_train), ws)]
    middle_window_train_expand = [middle_window_train[int(i // ws)] for i in range(len(waves_train))]
    middle_train = [np.average(waves_train[i:i+ws]) for i in range(len(waves_train) - ws + 1)]

    while len(middle_train) < len(waves_train):
        middle_train.append(middle_train[-1])

    sgn = 1
    if (np.argmax(waves_train[0:ws]) > np.argmin(waves_train[0:ws])):
        sgn = -1
 
    amplitudes_window_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(0, len(waves_train), ws)]
    amplitudes_window_train_expand = [amplitudes_window_train[int(i // ws)] for i in range(len(waves_train))]    
    amplitudes_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(len(waves_train) - ws + 1)]

    while len(amplitudes_train) < len(waves_train):
        amplitudes_train.append(amplitudes_train[-1])
 
    x_amplitudes, y_amplitudes = get_XY(amplitudes_train, ws_amplitude)

    x_amplitudes_to_predict = get_X(amplitudes_train, ws_amplitude)

    all_predicted_amplitudes = []
    
    for model_name in ["RNN"]: 
            
        if model_name == "RNN": 
            demo_model_amplitudes = create_RNN(hidden, ws_amplitude, (ws_amplitude, num_props), act_layer = "linear") 

        if model_name == "GRU": 
            demo_model_amplitudes = create_GRU(hidden, ws_amplitude, (ws_amplitude, num_props), act_layer = "linear") 

        if model_name == "LSTM": 
            demo_model_amplitudes = create_LSTM(hidden, ws_amplitude, (ws_amplitude, num_props), act_layer = "linear") 
            
        history_model_amplitudes = demo_model_amplitudes.fit(x_amplitudes, y_amplitudes, verbose = 1) 
   
        predict_train_amplitudes = demo_model_amplitudes.predict(x_amplitudes_to_predict)  
 
        print(math.sqrt(mean_squared_error(y_amplitudes, predict_train_amplitudes[:len(y_amplitudes)])) / (max(amplitudes_train) - min(amplitudes_train)))
       
        for i in range(len(predict_train_amplitudes)):
            for j in range(len(predict_train_amplitudes[i])):
                all_predicted_amplitudes.append(predict_train_amplitudes[i][j])

        plt.plot(amplitudes_train)
        plt.plot(all_predicted_amplitudes)
        plt.show()
        plt.close()
        
    x_waves, y_waves = get_XY(waves_train, ws_wave)

    x_waves_to_predict = get_X(waves_train, ws_wave)

    all_predicted_waves = []
    
    for model_name in ["RNN"]: 
            
        if model_name == "RNN": 
            demo_model_waves = create_RNN(hidden, ws_wave, (ws_wave, num_props), act_layer = "linear") 

        if model_name == "GRU": 
            demo_model_waves = create_GRU(hidden, ws_wave, (ws_wave, num_props), act_layer = "linear") 

        if model_name == "LSTM": 
            demo_model_waves = create_LSTM(hidden, ws_wave, (ws_wave, num_props), act_layer = "linear") 
            
        history_model_waves = demo_model_waves.fit(x_waves, y_waves, verbose = 1) 
   
        predict_train_waves = demo_model_waves.predict(x_waves_to_predict)  
 
        print(math.sqrt(mean_squared_error(y_waves, predict_train_waves[:len(y_waves)])) / (max(waves_train) - min(waves_train)))
       
        for i in range(len(predict_train_waves)):
            for j in range(len(predict_train_waves[i])):
                all_predicted_waves.append(predict_train_waves[i][j])

        plt.plot(waves_train)
        plt.plot(all_predicted_waves)
        plt.show()
        plt.close()
        
    x_middles, y_middles = get_XY(middle_train, ws_middle)

    x_middles_to_predict = get_X(middle_train, ws_middle)

    all_predicted_middles = []
    
    for model_name in ["RNN"]: 
            
        if model_name == "RNN": 
            demo_model_middles = create_RNN(hidden, ws_middle, (ws_middle, num_props), act_layer = "linear") 

        if model_name == "GRU": 
            demo_model_middles = create_GRU(hidden, ws_middle, (ws_middle, num_props), act_layer = "linear") 

        if model_name == "LSTM": 
            demo_model_middles = create_LSTM(hidden, ws_middle, (ws_middle, num_props), act_layer = "linear") 
            
        history_model_middles = demo_model_middles.fit(x_middles, y_middles, verbose = 1) 
   
        predict_train_middles = demo_model_middles.predict(x_middles_to_predict)  
 
        print(math.sqrt(mean_squared_error(y_middles, predict_train_middles[:len(y_middles)])) / (max(middle_train) - min(middle_train)))
       
        for i in range(len(predict_train_middles)):
            for j in range(len(predict_train_middles[i])):
                all_predicted_middles.append(predict_train_middles[i][j])

        plt.plot(middle_train)
        plt.plot(all_predicted_middles)
        plt.show()
        plt.close()

    break