import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ws = 365

deg_amplitudes = 1

deg_middle = 1

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
      
    poly_amplitudes = [0, np.min(amplitudes_train)]
    amplitudes_extrapolate = [np.sum([poly_amplitudes[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes))]) for x in range(len(waves) + new_delta.days)]
    amplitudes_predicted_all = amplitudes_extrapolate[:len(waves)] 
        
    poly_amplitudes_window = [0, np.min(amplitudes_window_train)]
    amplitudes_window_extrapolate = [np.sum([poly_amplitudes_window[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes_window))]) for x in range((len(waves) + new_delta.days) // ws + 1)]
    amplitudes_window_extrapolate = [np.average(amplitudes_extrapolate[i:i+ws]) for i in range(0, len(waves) + new_delta.days, ws)]
    amplitudes_window_predicted_all = amplitudes_window_extrapolate[:len(waves) // ws + 1] 
  
    poly_middle = [0.0036 / 365, np.min(middle_train)]
    middle_extrapolate = [np.sum([poly_middle[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle))]) for x in range(len(waves) + new_delta.days)]
    middle_predicted_all = middle_extrapolate[:len(waves)] 
 
    poly_middle_window = [0.0036, min(middle_window_train)]
    middle_window_extrapolate = [np.sum([poly_middle_window[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle_window))]) for x in range((len(waves) + new_delta.days) // ws + 1)]
    middle_window_predicted_all = middle_window_extrapolate[:len(waves) // ws + 1] 

    sinus_like_extrapolate = [middle_extrapolate[x] + sgn * amplitudes_extrapolate[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves) + new_delta.days)]
    sinus_like_predicted_all = sinus_like_extrapolate[:len(waves)]
    sinus_like_predicted_train = sinus_like_predicted_all[:len(waves_train)]
    sinus_like_predicted_test = sinus_like_predicted_all[len(waves_train):]
  
    rmse_estimate_train = math.sqrt(mean_squared_error(waves_train, sinus_like_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_test = math.sqrt(mean_squared_error(waves_test, sinus_like_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_all = math.sqrt(mean_squared_error(waves, sinus_like_predicted_all)) / (max(waves) - min(waves))
    
    print(rmse_estimate_train, rmse_estimate_test, rmse_estimate_all)
    
    middle_window_extrapolate_expand = [middle_window_extrapolate[int(i // ws)] for i in range(len(waves) + new_delta.days)]
 
    amplitudes_window_extrapolate_expand = [amplitudes_window_extrapolate[int(i // ws)] for i in range(len(waves) + new_delta.days)]
 
    sinus_like_window_extrapolate = [middle_window_extrapolate_expand[x] + sgn * amplitudes_window_extrapolate_expand[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves) + new_delta.days)]
    sinus_like_window_predicted_all = sinus_like_window_extrapolate[:len(waves)]
    sinus_like_window_predicted_train = sinus_like_window_predicted_all[:len(waves_train)]
    sinus_like_window_predicted_test = sinus_like_window_predicted_all[len(waves_train):]
  
    rmse_estimate_window_train = math.sqrt(mean_squared_error(waves_train, sinus_like_window_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_window_test = math.sqrt(mean_squared_error(waves_test, sinus_like_window_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_window_all = math.sqrt(mean_squared_error(waves, sinus_like_window_predicted_all)) / (max(waves) - min(waves))

    print(rmse_estimate_window_train, rmse_estimate_window_test, rmse_estimate_window_all)

    plt.figure(figsize=(16, 9))
    plt.plot(waves) 
    plt.plot(sinus_like_extrapolate, color = "red") 
    plt.plot(amplitudes_train, color = "green")
    plt.plot(amplitudes_extrapolate, color = "green") 
    plt.plot(middle_train, color = "blue")
    plt.plot(middle_extrapolate, color = "blue")
    plt.axvline(len(waves_train), color = "orange")
    plt.axvline(len(waves), color = "orange")
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.show()
    plt.close() 

    plt.figure(figsize=(16, 9))
    plt.plot(waves) 
    plt.plot(sinus_like_window_extrapolate, color = "red") 
    plt.plot(amplitudes_window_train_expand, color = "green")
    plt.plot(amplitudes_window_extrapolate_expand, color = "green") 
    plt.plot(middle_window_train_expand, color = "blue")
    plt.plot(middle_window_extrapolate_expand, color = "blue")
    plt.axvline(len(waves_train), color = "orange")
    plt.axvline(len(waves), color = "orange")
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.show()
    plt.close() 