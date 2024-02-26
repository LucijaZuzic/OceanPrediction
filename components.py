import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utilities import get_XY, create_GRU, create_LSTM, create_RNN, print_predictions

ws = 365
models = ["LSTM", "RNN", "GRU"] 
hidden = 200
num_props = 1
ws_amplitude = 10
ws_middle = 10

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    waves = list(file_data["sla"]) 

    waves_train = list(file_data["sla"][:int(0.7 * len(waves) // ws * ws)])

    waves_test = list(file_data["sla"][int(0.7 * len(waves) // ws * ws):])
    
    middle_train = [np.average(waves_train[i:i+ws]) for i in range(0, len(waves_train), ws)]

    sgn = 1
    if (np.argmax(waves_train[0:ws]) > np.argmin(waves_train[0:ws])):
        sgn = -1
 
    amplitudes_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[int(i // ws)], middle_train[int(i // ws)] - np.min(waves_train[i:i+ws])]) for i in range(0, len(waves_train), ws)]   
  
    sinus_like_train = [middle_train[int(x // ws)] + sgn * amplitudes_train[int(x // ws)] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves_train))] 

    rmse_train = math.sqrt(mean_squared_error(waves_train, sinus_like_train)) / (max(waves) - min(waves)) 
 
    print(np.argmax(waves_train[0:ws]), np.argmin(waves_train[0:ws]), sgn, rmse_train) 
 
    amplitudes_long = [amplitudes_train[int(x // ws)] for x in range(len(waves_train))]  
    middle_long = [middle_train[int(x // ws)] for x in range(len(waves_train))] 

    plt.plot(waves_train)
    plt.plot(sinus_like_train) 
    plt.plot(middle_long) 
    plt.plot(amplitudes_long)
    plt.show()
    plt.close()

    x_amplitudes, y_amplitudes = get_XY(amplitudes_train, ws_amplitude)
    data_last_window = amplitudes_train[-ws_amplitude:]
    for ix in range(len(data_last_window)):
        data_last_window.append(0)
    
    for model_name in models: 
            
        if model_name == "RNN": 
            demo_model_amplitudes = create_RNN(hidden, ws_amplitude, (ws_amplitude, num_props), act_layer = "custom_activation_abs") 

        if model_name == "GRU": 
            demo_model_amplitudes = create_GRU(hidden, ws_amplitude, (ws_amplitude, num_props), act_layer = "custom_activation_abs") 

        if model_name == "LSTM": 
            demo_model_amplitudes = create_LSTM(hidden, ws_amplitude, (ws_amplitude, num_props), act_layer = "custom_activation_abs") 
            
        history_model_amplitudes = demo_model_amplitudes.fit(x_amplitudes, y_amplitudes, verbose = 1) 
 
        x_amplitudes_last_window, y_amplitudes_last_window = get_XY(data_last_window, ws_amplitude)

        predict_train_amplitudes = demo_model_amplitudes.predict(x_amplitudes) 

        predict_test_amplitudes = demo_model_amplitudes.predict(x_amplitudes_last_window) 

        print(x_amplitudes)
        print(predict_train_amplitudes)
        print(predict_test_amplitudes)

    amplitudes_test = []
    for ix in range(len(middle_train) + len(range(0, len(waves_test), ws))):
        amplitudes_test.append(amplitudes_train[-1])
    for ix in range(len(amplitudes_train)):
        amplitudes_test[ix] = amplitudes_train[ix]
      
    deg_middle = 1
    poly_middle = np.polyfit(range(len(middle_train)), middle_train, deg_middle) 
    middle_estimate = [np.sum([poly_middle[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle))]) for x in range(len(middle_train))] 
     
    rmse_middle = math.sqrt(mean_squared_error(middle_train, middle_estimate)) / (max(middle_train) - min(middle_train)) 
    print(rmse_middle) 

    middle_predicted = [np.sum([poly_middle[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle))]) for x in range(len(middle_train) + len(range(0, len(waves_test), ws)))] 
    middle_test = []
    for ix in range(len(middle_predicted)):
        middle_test.append(middle_predicted[ix])
    for ix in range(len(middle_train)):
        middle_test[ix] = middle_train[ix]
    
    plt.plot(middle_train) 
    plt.plot(middle_predicted)
    plt.show()
    plt.close()

    sinus_like_test = [middle_test[int(x // ws)] + sgn * amplitudes_test[int(x // ws)] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves_train), len(waves_train) + len(waves_test))] 
 
    rmse_test = math.sqrt(mean_squared_error(waves_test, sinus_like_test)) / (max(waves) - min(waves)) 
    print(rmse_test) 

    amplitudes_predicted_long = [amplitudes_test[int(x // ws)] for x in range(len(waves_train), len(waves_train) + len(waves_test))]  
    middle_predicted_long = [middle_test[int(x // ws)] for x in range(len(waves_train), len(waves_train) + len(waves_test))]  

    plt.plot(waves_test)
    plt.plot(sinus_like_test)
    plt.plot(amplitudes_predicted_long) 
    plt.plot(middle_predicted_long)
    plt.show()
    plt.close()

    plt.plot(range(len(waves_train)), waves_train)
    plt.plot(range(len(waves_train)), sinus_like_train) 
    plt.plot(range(len(waves_train)), middle_long) 
    plt.plot(range(len(waves_train)), amplitudes_long)
    plt.plot(range(len(waves_train), len(waves_train) + len(waves_test)), waves_test)
    plt.plot(range(len(waves_train), len(waves_train) + len(waves_test)), sinus_like_test)
    plt.plot(range(len(waves_train), len(waves_train) + len(waves_test)), amplitudes_predicted_long) 
    plt.plot(range(len(waves_train), len(waves_train) + len(waves_test)), middle_predicted_long)
    plt.show()
    plt.close()

    print(len(amplitudes_train), len(amplitudes_test))