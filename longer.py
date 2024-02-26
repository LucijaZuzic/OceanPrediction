import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

ws = 365

deg_amplitudes = 1

deg_middle = 1

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    waves = list(file_data["sla"]) 

    waves_train = list(file_data["sla"][:int(0.7 * len(waves) // ws * ws)])

    waves_test = list(file_data["sla"][int(0.7 * len(waves) // ws * ws):])
    
    middle_window_train = [np.average(waves_train[i:i+ws]) for i in range(0, len(waves_train), ws)]
    middle_window_train_expand = [middle_window_train[int(i // ws)] for i in range(len(waves_train))]
    middle_train = [np.average(waves_train[i:i+ws]) for i in range(len(waves_train))]

    sgn = 1
    if (np.argmax(waves_train[0:ws]) > np.argmin(waves_train[0:ws])):
        sgn = -1
 
    amplitudes_window_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(0, len(waves_train), ws)]
    amplitudes_window_train_expand = [amplitudes_window_train[int(i // ws)] for i in range(len(waves_train))]    
    amplitudes_train = [np.min([np.max(waves_train[i:i+ws]) - middle_train[i], middle_train[i] - np.min(waves_train[i:i+ws])]) for i in range(len(waves_train))]
  
    sinus_like_train = [middle_train[x] + sgn * amplitudes_train[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves_train))]
    sinus_like_window_train = [middle_window_train_expand[x] + sgn * amplitudes_window_train_expand[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves_train))]

    rmse_train = math.sqrt(mean_squared_error(waves_train, sinus_like_train)) / (max(waves) - min(waves))
    rmse_window_train = math.sqrt(mean_squared_error(waves_train, sinus_like_window_train)) / (max(waves) - min(waves))

    poly_amplitudes = np.polyfit(range(len(amplitudes_train)), amplitudes_train, deg_amplitudes)
    amplitudes_predicted_all = [np.sum([poly_amplitudes[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes))]) for x in range(len(waves))]
    amplitudes_predicted_train = amplitudes_predicted_all[:len(amplitudes_train)]
    amplitudes_predicted_test = amplitudes_predicted_all[len(amplitudes_train):]
     
    rmse_amplitudes = math.sqrt(mean_squared_error(amplitudes_train, amplitudes_predicted_train)) / (max(amplitudes_train) - min(amplitudes_train))
 
    amplitudes_estimate_predicted = []
    for ix in range(len(amplitudes_predicted_all)):
        amplitudes_estimate_predicted.append(amplitudes_predicted_all[ix])
    for ix in range(len(amplitudes_train)):
        amplitudes_estimate_predicted[ix] = amplitudes_train[ix]

    poly_amplitudes_window = np.polyfit(range(len(amplitudes_window_train)), amplitudes_window_train, deg_amplitudes)
    amplitudes_window_predicted_all = [np.sum([poly_amplitudes_window[ix] * x ** (deg_amplitudes - ix) for ix in range(len(poly_amplitudes_window))]) for x in range(len(waves) // ws + 1)]
    amplitudes_window_predicted_train = amplitudes_window_predicted_all[:len(amplitudes_window_train)]
    amplitudes_window_predicted_test = amplitudes_window_predicted_all[len(amplitudes_window_train):]
     
    rmse_amplitudes_window = math.sqrt(mean_squared_error(amplitudes_window_train, amplitudes_window_predicted_train)) / (max(amplitudes_window_train) - min(amplitudes_window_train))
 
    amplitudes_window_estimate_predicted = []
    for ix in range(len(amplitudes_window_predicted_all)):
        amplitudes_window_estimate_predicted.append(amplitudes_window_predicted_all[ix])
    for ix in range(len(amplitudes_window_train)):
        amplitudes_window_estimate_predicted[ix] = amplitudes_window_train[ix]
 
    poly_middle = np.polyfit(range(len(middle_train)), middle_train, deg_middle) 
    middle_predicted_all = [np.sum([poly_middle[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle))]) for x in range(len(waves))]
    middle_predicted_train = middle_predicted_all[:len(middle_train)]
    middle_predicted_test = middle_predicted_all[len(middle_train):]
     
    rmse_middle = math.sqrt(mean_squared_error(middle_train, middle_predicted_train)) / (max(middle_train) - min(middle_train))
 
    middle_estimate_predicted = []
    for ix in range(len(middle_predicted_all)):
        middle_estimate_predicted.append(middle_predicted_all[ix])
    for ix in range(len(middle_train)):
        middle_estimate_predicted[ix] = middle_train[ix]

    poly_middle_window = np.polyfit(range(len(middle_window_train)), middle_window_train, deg_middle)
    middle_window_predicted_all = [np.sum([poly_middle_window[ix] * x ** (deg_middle - ix) for ix in range(len(poly_middle_window))]) for x in range(len(waves) // ws + 1)]
    middle_window_predicted_train = middle_window_predicted_all[:len(middle_window_train)]
    middle_window_predicted_test = middle_window_predicted_all[len(middle_window_train):]
     
    rmse_middle_window = math.sqrt(mean_squared_error(middle_window_train, middle_window_predicted_train)) / (max(middle_window_train) - min(middle_window_train))
 
    middle_window_estimate_predicted = []
    for ix in range(len(middle_window_predicted_all)):
        middle_window_estimate_predicted.append(middle_window_predicted_all[ix])
    for ix in range(len(middle_window_train)):
        middle_window_estimate_predicted[ix] = middle_window_train[ix]
 
    sinus_like_predicted_all = [middle_predicted_all[x] + sgn * amplitudes_predicted_all[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves))]
    middle_predicted_test = middle_predicted_all[-len(waves_test):]
    sinus_like_predicted_train = sinus_like_predicted_all[:len(waves_train)]
    sinus_like_predicted_test = sinus_like_predicted_all[len(waves_train):]
    sinus_like_estimate_predicted = []
    for ix in range(len(sinus_like_predicted_all)):
        sinus_like_estimate_predicted.append(sinus_like_predicted_all[ix])
    for ix in range(len(sinus_like_train)):
        sinus_like_estimate_predicted[ix] = sinus_like_train[ix]

    rmse_estimate_train = math.sqrt(mean_squared_error(waves_train, sinus_like_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_test = math.sqrt(mean_squared_error(waves_test, sinus_like_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_all = math.sqrt(mean_squared_error(waves, sinus_like_predicted_all)) / (max(waves) - min(waves))
    print(rmse_train, rmse_amplitudes, rmse_middle, rmse_estimate_train, rmse_estimate_test, rmse_estimate_all)
    
    middle_window_predicted_expand_all = [middle_window_predicted_all[int(i // ws)] for i in range(len(waves))]
    middle_window_estimate_predicted_expand = [middle_window_estimate_predicted[int(i // ws)] for i in range(len(waves))]
    middle_window_predicted_expand_train = middle_window_predicted_expand_all[:len(waves_train)]
    middle_window_predicted_expand_test = middle_window_predicted_expand_all[len(waves_train):]

    amplitudes_window_predicted_expand_all = [amplitudes_window_predicted_all[int(i // ws)] for i in range(len(waves))]
    amplitudes_window_estimate_predicted_expand = [amplitudes_window_estimate_predicted[int(i // ws)] for i in range(len(waves))]
    amplitudes_window_predicted_expand_train = amplitudes_window_predicted_expand_all[:len(waves_train)]
    amplitudes_window_predicted_expand_test = amplitudes_window_predicted_expand_all[len(waves_train):]
        
    sinus_like_window_predicted_all = [middle_window_predicted_expand_all[x] + sgn * amplitudes_window_predicted_expand_all[x] * np.sin(x * 2 * np.pi / ws) for x in range(len(waves))]
    middle_predicted_test = middle_predicted_all[-len(waves_test):]
    sinus_like_window_predicted_train = sinus_like_window_predicted_all[:len(waves_train)]
    sinus_like_window_predicted_test = sinus_like_window_predicted_all[len(waves_train):]
    sinus_like_window_estimate_predicted = []
    for ix in range(len(sinus_like_window_predicted_all)):
        sinus_like_window_estimate_predicted.append(sinus_like_window_predicted_all[ix])
    for ix in range(len(sinus_like_window_train)):
        sinus_like_window_estimate_predicted[ix] = sinus_like_window_train[ix]
    
    rmse_estimate_window_train = math.sqrt(mean_squared_error(waves_train, sinus_like_window_predicted_train)) / (max(waves) - min(waves))
    rmse_estimate_window_test = math.sqrt(mean_squared_error(waves_test, sinus_like_window_predicted_test)) / (max(waves) - min(waves))
    rmse_estimate_window_all = math.sqrt(mean_squared_error(waves, sinus_like_window_predicted_all)) / (max(waves) - min(waves))
    print(rmse_window_train, rmse_amplitudes_window, rmse_middle_window, rmse_estimate_window_train, rmse_estimate_window_test, rmse_estimate_window_all)

    plt.plot(waves)
    plt.plot(sinus_like_estimate_predicted, color = "red")
    plt.plot(amplitudes_estimate_predicted, color = "green")
    plt.plot(middle_estimate_predicted, color = "blue")
    plt.axvline(len(waves_train), color = "orange")
    plt.show()
    plt.close()
    
    plt.plot(waves)
    plt.plot(sinus_like_window_estimate_predicted, color = "red")
    plt.plot(amplitudes_window_estimate_predicted_expand, color = "green")
    plt.plot(middle_window_estimate_predicted_expand, color = "blue")
    plt.axvline(len(waves_train), color = "orange")
    plt.show()
    plt.close()