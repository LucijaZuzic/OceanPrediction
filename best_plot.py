import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

ws = 365

def est_plot(filename, method_amplitudes_first, method_amplitudes, method_middle):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")
     
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
 
    if method_amplitudes_first == "min":
        amplitudes_window_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(0, len(waves_train), ws)]
        amplitudes_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(len(waves_train) - ws + 1)]

    if method_amplitudes_first == "max":
        amplitudes_window_train = [np.max([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(0, len(waves_train), ws)]
        amplitudes_train = [np.max([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(len(waves_train) - ws + 1)]

    if method_amplitudes_first == "avg":
        amplitudes_window_train = [np.average([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(0, len(waves_train), ws)]
        amplitudes_train = [np.average([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(len(waves_train) - ws + 1)]

    while len(amplitudes_train) < len(waves_train):
        amplitudes_train.append(amplitudes_train[-1]) 

    amplitudes_window_train_expand = [amplitudes_window_train[int(i // ws)] for i in range(len(waves_train))]    

    deg_amplitudes = 1
    
    if method_amplitudes == "min":
        poly_amplitudes = [0, np.min(amplitudes_train)]

    if method_amplitudes == "max":
        poly_amplitudes = [0, np.max(amplitudes_train)]

    if method_amplitudes == "avg":
        poly_amplitudes = [0, np.average(amplitudes_train)]

    if method_amplitudes == "poly1d":
        poly_amplitudes = np.polyfit(range(len(amplitudes_train)), amplitudes_train, deg_amplitudes)

    if method_amplitudes == "poly2d":
        deg_amplitudes = 2
        poly_amplitudes = np.polyfit(range(len(amplitudes_train)), amplitudes_train, deg_amplitudes)

    amplitudes_extrapolate = [np.sum([poly_amplitudes[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes))]) for x in range(len(waves) + new_delta.days)]
    amplitudes_predicted_all = amplitudes_extrapolate[:len(waves)] 
         
    deg_amplitudes = 1

    if method_amplitudes == "min":
        poly_amplitudes_window = [0, np.min(amplitudes_window_train)]
        
    if method_amplitudes == "max":
        poly_amplitudes_window = [0, np.max(amplitudes_window_train)]

    if method_amplitudes == "avg":
        poly_amplitudes_window = [0, np.average(amplitudes_window_train)]

    if method_amplitudes == "poly1d":
        poly_amplitudes_window = np.polyfit(range(len(amplitudes_window_train)), amplitudes_window_train, deg_amplitudes)

    if method_amplitudes == "poly2d":
        deg_amplitudes = 2
        poly_amplitudes_window = np.polyfit(range(len(amplitudes_window_train)), amplitudes_window_train, deg_amplitudes)
  
    amplitudes_window_extrapolate = [np.sum([poly_amplitudes_window[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes_window))]) for x in range(len(waves) + new_delta.days)]
    amplitudes_window_predicted_all = amplitudes_window_extrapolate[:len(waves)] 

    deg_middle = 1

    if method_middle == "min":
        poly_middle = [0, np.min(middle_train)]

    if method_middle == "max":
        poly_middle = [0, np.max(middle_train)]

    if method_middle == "avg":
        poly_middle = [0, np.average(middle_train)]

    if method_middle == "poly1d":
        poly_middle = np.polyfit(range(len(middle_train)), middle_train, deg_middle)

    if method_middle == "poly2d":
        deg_middle = 2
        poly_middle = np.polyfit(range(len(middle_train)), middle_train, deg_middle)
  
    middle_extrapolate = [np.sum([poly_middle[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle))]) for x in range(len(waves) + new_delta.days)]
    middle_predicted_all = middle_extrapolate[:len(waves)] 

    deg_middle = 1

    if method_middle == "min":
        poly_middle_window = [0, np.min(middle_window_train)]

    if method_middle == "max":
        poly_middle_window = [0, np.max(middle_window_train)]

    if method_middle == "avg":
        poly_middle_window = [0, np.average(middle_window_train)]

    if method_middle == "poly1d":
        poly_middle_window = np.polyfit(range(len(middle_window_train)), middle_window_train, deg_middle)

    if method_middle == "poly2d":
        deg_middle = 2
        poly_middle_window = np.polyfit(range(len(middle_window_train)), middle_window_train, deg_middle)

    middle_window_extrapolate = [np.sum([poly_middle_window[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle_window))]) for x in range((len(waves) + new_delta.days) // ws + 1)]
    middle_window_predicted_all = middle_window_extrapolate[:len(waves) // ws + 1] 
    
    sinus_like_extrapolate = [middle_extrapolate[x] + sgn * amplitudes_extrapolate[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves) + new_delta.days)]
    sinus_like_predicted_all = sinus_like_extrapolate[:len(waves)]
    sinus_like_predicted_train = sinus_like_predicted_all[:len(waves_train)]
    sinus_like_predicted_test = sinus_like_predicted_all[len(waves_train):]
  
    rmse_estimate_train = math.sqrt(mean_squared_error(waves_train, sinus_like_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_test = math.sqrt(mean_squared_error(waves_test, sinus_like_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_all = math.sqrt(mean_squared_error(waves, sinus_like_predicted_all)) / (max(waves) - min(waves))
     
    middle_window_extrapolate_expand = [middle_window_extrapolate[int(i // ws)] for i in range(len(waves) + new_delta.days)] 
    amplitudes_window_extrapolate_expand = [amplitudes_window_extrapolate[int(i // ws)] for i in range(len(waves) + new_delta.days)]
  
    sinus_like_window_extrapolate = [middle_window_extrapolate_expand[x] + sgn * amplitudes_window_extrapolate_expand[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves) + new_delta.days)]
    sinus_like_window_predicted_all = sinus_like_window_extrapolate[:len(waves)]
    sinus_like_window_predicted_train = sinus_like_window_predicted_all[:len(waves_train)]
    sinus_like_window_predicted_test = sinus_like_window_predicted_all[len(waves_train):]
  
    rmse_estimate_window_train = math.sqrt(mean_squared_error(waves_train, sinus_like_window_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_window_test = math.sqrt(mean_squared_error(waves_test, sinus_like_window_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_window_all = math.sqrt(mean_squared_error(waves, sinus_like_window_predicted_all)) / (max(waves) - min(waves))
 
    print(rmse_estimate_train, rmse_estimate_test, rmse_estimate_all, rmse_estimate_window_train, rmse_estimate_window_test, rmse_estimate_window_all)
    print(rmse_estimate_train > rmse_estimate_window_train, rmse_estimate_test > rmse_estimate_window_test, rmse_estimate_all > rmse_estimate_window_all)

    key = "_af_" + method_amplitudes_first + "_a_" + method_amplitudes + "_m_" + method_middle
    name_w = "method_plots/" + filename.replace(".csv", "") + key + "_w.png"
    name_nw = "method_plots/" + filename.replace(".csv", "") + key + ".png"

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
    plt.savefig(name_nw, bbox_inches = "tight")
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
    plt.savefig(name_w, bbox_inches = "tight")
    plt.close()

if not os.path.isdir("method_plots"):
    os.makedirs("method_plots")

for filename in os.listdir("method_scores"):  

    file_data = pd.read_csv("method_scores/" + filename, index_col = False, sep = ";")
    
    best = (-1, 1000)
    best_all = (-1, 1000)
    best_train = (-1, 1000)
    best_test = (-1, 1000)

    best_nw_all = (-1, 1000)
    best_nw_train = (-1, 1000)
    best_nw_test = (-1, 1000)

    best_w_all = (-1, 1000)
    best_w_train = (-1, 1000)
    best_w_test = (-1, 1000)

    for val in file_data.columns[1:]:

        if float(file_data[val][0]) < best[1]:
            best = (val, float(file_data[val][0]))

        if "train" in val: 
            if float(file_data[val][0]) < best_train[1]:
                best_train = (val, float(file_data[val][0]))
            if "w" in val: 
                if float(file_data[val][0]) < best_w_train[1]:
                    best_w_train = (val, float(file_data[val][0]))
            else: 
                if float(file_data[val][0]) < best_nw_train[1]:
                    best_nw_train = (val, float(file_data[val][0]))
                
        if "test" in val: 
            if float(file_data[val][0]) < best_test[1]:
                best_test = (val, float(file_data[val][0]))
            if "w" in val: 
                if float(file_data[val][0]) < best_w_test[1]:
                    best_w_test = (val, float(file_data[val][0]))
            else: 
                if float(file_data[val][0]) < best_nw_test[1]:
                    best_nw_test = (val, float(file_data[val][0]))
                
        if "all" in val: 
            if float(file_data[val][0]) < best_all[1]:
                best_all = (val, float(file_data[val][0]))
            if "w" in val: 
                if float(file_data[val][0]) < best_w_all[1]:
                    best_w_all = (val, float(file_data[val][0]))
            else: 
                if float(file_data[val][0]) < best_nw_all[1]:
                    best_nw_all = (val, float(file_data[val][0]))

    key_best = tuple(best[0].split("_")[1:7:2])

    key_best_train = tuple(best_train[0].split("_")[1:7:2])
    key_best_w_train = tuple(best_w_train[0].split("_")[1:7:2])
    key_best_nw_train = tuple(best_nw_train[0].split("_")[1:7:2])
 
    key_best_test = tuple(best_test[0].split("_")[1:7:2])
    key_best_w_test = tuple(best_w_test[0].split("_")[1:7:2])
    key_best_nw_test = tuple(best_nw_test[0].split("_")[1:7:2])

    key_best_all = tuple(best_all[0].split("_")[1:7:2])
    key_best_w_all = tuple(best_w_all[0].split("_")[1:7:2])
    key_best_nw_all = tuple(best_nw_all[0].split("_")[1:7:2])
    
    set_keys = set({key_best, key_best_train, key_best_w_train, key_best_nw_train, key_best_test, key_best_w_test, key_best_nw_test, key_best_all, key_best_w_all, key_best_nw_all})

    best_key = ("min", "min", "poly2d")

    if best_key not in set_keys:
        set_keys.add(best_key)

    print(set_keys)

    for key_set in set_keys:
        est_plot(filename, key_set[0], key_set[1], key_set[2])