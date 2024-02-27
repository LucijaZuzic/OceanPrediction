import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error 

ws = 365

def est_data(filename, method_amplitudes_first, method_amplitudes, method_middle):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")
     
    waves = list(file_data["sla"]) 
                            
    waves_train = list(file_data["sla"][:int(0.7 * len(waves) // ws * ws)])

    waves_test = list(file_data["sla"][int(0.7 * len(waves) // ws * ws):])
    
    middle_window_train = [np.average(waves_train[i:i+ws]) for i in range(0, len(waves_train), ws)] 
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

    amplitudes_extrapolate = [np.sum([poly_amplitudes[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes))]) for x in range(len(waves))]
         
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
 
    amplitudes_window_extrapolate = [np.sum([poly_amplitudes_window[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes_window))]) for x in range(len(waves) // ws + 1)]
  
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
 
    middle_extrapolate = [np.sum([poly_middle[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle))]) for x in range(len(waves))]
  
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

    middle_window_extrapolate = [np.sum([poly_middle_window[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle_window))]) for x in range((len(waves)) // ws + 1)]
 
    sinus_like_predicted_all = [middle_extrapolate[x] + sgn * amplitudes_extrapolate[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves))]
    sinus_like_predicted_train = sinus_like_predicted_all[:len(waves_train)]
    sinus_like_predicted_test = sinus_like_predicted_all[len(waves_train):]
  
    rmse_estimate_train = math.sqrt(mean_squared_error(waves_train, sinus_like_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_test = math.sqrt(mean_squared_error(waves_test, sinus_like_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_all = math.sqrt(mean_squared_error(waves, sinus_like_predicted_all)) / (max(waves) - min(waves))
     
    middle_window_extrapolate_expand = [middle_window_extrapolate[int(i // ws)] for i in range(len(waves))]
 
    amplitudes_window_extrapolate_expand = [amplitudes_window_extrapolate[int(i // ws)] for i in range(len(waves))]
 
    sinus_like_window_predicted_all = [middle_window_extrapolate_expand[x] + sgn * amplitudes_window_extrapolate_expand[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves))]
    sinus_like_window_predicted_train = sinus_like_window_predicted_all[:len(waves_train)]
    sinus_like_window_predicted_test = sinus_like_window_predicted_all[len(waves_train):]
  
    rmse_estimate_window_train = math.sqrt(mean_squared_error(waves_train, sinus_like_window_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_window_test = math.sqrt(mean_squared_error(waves_test, sinus_like_window_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_window_all = math.sqrt(mean_squared_error(waves, sinus_like_window_predicted_all)) / (max(waves) - min(waves))

    return {
        "w_train": rmse_estimate_window_train,
        "w_test": rmse_estimate_window_test,
        "w_all": rmse_estimate_window_all,
        "train": rmse_estimate_train,
        "test": rmse_estimate_test,
        "all": rmse_estimate_all
    }

dict_files = dict()
dict_files_all = dict()
dict_files_train = dict()
dict_files_test = dict()

if not os.path.isdir("method_scores"):
    os.makedirs("method_scores")

for filename in os.listdir("processed"): 

    dict_files[filename] = dict()
    dict_files_all[filename] = dict()
    dict_files_train[filename] = dict()
    dict_files_test[filename] = dict()

    for method_amplitudes_first in ["min", "max", "avg"]:
        for method_amplitudes in ["min", "max", "avg", "poly1d", "poly2d"]:
            for method_middle in ["min", "max", "avg", "poly1d", "poly2d"]:

                retvals = est_data(filename, method_amplitudes_first, method_amplitudes, method_middle)

                key = "af_" + method_amplitudes_first + "_a_" + method_amplitudes + "_m_" + method_middle

                for val in retvals:

                    dict_files[filename][key + "_" + val] = retvals[val]
      
    strpr = "filename;"
    for key in dict_files[filename]:
        strpr += key + ";"
    strpr = strpr[:-1] + "\n" 
    strpr += filename + ";"
    for key in dict_files[filename]:
        strpr += str(dict_files[filename][key]) + ";"
    strpr = strpr[:-1] + "\n"

    file_open = open("method_scores/" + filename, "w")
    file_open.write(strpr)
    file_open.close()