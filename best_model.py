import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

ws = 365

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    waves = list(file_data["sla"]) 

    waves_train = list(file_data["sla"][:int(0.7 * len(waves) // ws * ws)])

    waves_test = list(file_data["sla"][int(0.7 * len(waves) // ws * ws):])
    
    mini_points = []

    maxi_points = []
     
    for i in range(len(waves_train)):
        
        window = list(waves_train[i: i + ws])

        mini_point = min(window)

        mini_index = window.index(mini_point)

        mini_points.append((i + mini_index, mini_point))

    mini_points = sorted(list(set(mini_points)))

    mini_points_group = [[(mini_points[0][1], mini_points[0][0])]]

    for i in range(1, len(mini_points)): 

        ix_diff_mini_point = mini_points[i][0] - mini_points[i - 1][0]

        if ix_diff_mini_point >= ws // 2:

            mini_points_group.append([])
        
        mini_points_group[-1].append((mini_points[i][1], mini_points[i][0]))
  
    mini_points_new_x = []

    mini_points_new_y = []

    for mini_group_index in range(len(mini_points_group)):

        mini_points_group[mini_group_index] = sorted(mini_points_group[mini_group_index])[0]

        mini_points_new_x.append(mini_points_group[mini_group_index][1])

        mini_points_new_y.append(mini_points_group[mini_group_index][0])

    amplitudes = []

    offsets = [] 

    periods = [] 

    midpoints = []

    x_vals = []  

    for x_val in range(mini_points_new_x[0]):

        x_vals.append(x_val)

        midpoints.append((mini_points_new_y[0] + waves_train[0]) / 2)

        amplitudes.append((abs(mini_points_new_y[0] - midpoints[-1]) + abs(waves_train[0] - midpoints[-1])) / 2)
            
        periods.append(2 * mini_points_new_x[0])

        offsets.append(0)

    maxi_points_x = []

    maxi_points_y = []

    left_points_x = []

    left_points_y = []

    right_points_x = []

    right_points_y = []

    for i in range(len(mini_points_new_x) - 1): 

        window = list(waves_train[mini_points_new_x[i]:mini_points_new_x[i + 1]])

        maxi_y = max(window)

        maxi_x = mini_points_new_x[i] + window.index(maxi_y) 

        maxi_points_x.append(maxi_x)

        maxi_points_y.append(maxi_y)

        left_points_x.append((mini_points_new_x[i] + maxi_x) / 2)

        left_points_y.append((mini_points_new_y[i] + maxi_y) / 2)

        right_points_x.append((mini_points_new_x[i + 1] + maxi_x) / 2)

        right_points_y.append((mini_points_new_y[i + 1] + maxi_y) / 2)

        for x_val in range(mini_points_new_x[i], maxi_x):

            x_vals.append(x_val)

            midpoints.append((mini_points_new_y[i] + maxi_y) / 2)

            amplitudes.append(- (abs(mini_points_new_y[i] - midpoints[-1]) + abs(maxi_y - midpoints[-1])) / 2)
                
            periods.append(2 * (maxi_x - mini_points_new_x[i]))

            offsets.append(- mini_points_new_x[i] / periods[-1] * 2 * np.pi)

        for x_val in range(maxi_x, mini_points_new_x[i + 1]):

            x_vals.append(x_val)

            midpoints.append((mini_points_new_y[i + 1] + maxi_y) / 2)

            amplitudes.append((abs(mini_points_new_y[i + 1] - midpoints[-1]) + abs(maxi_y - midpoints[-1])) / 2)
            
            periods.append(2 * (mini_points_new_x[i + 1] - maxi_x))

            offsets.append(- maxi_x / periods[-1] * 2 * np.pi)

    for x_val in range(mini_points_new_x[-1], len(waves_train)):

        x_vals.append(x_val)

        midpoints.append((mini_points_new_y[-1] + waves_train[-1]) / 2)

        amplitudes.append(- (abs(mini_points_new_y[-1] - midpoints[-1]) + abs(waves_train[-1] - midpoints[-1])) / 2)
            
        periods.append(2 * (len(waves_train) - mini_points_new_x[-1]))

        offsets.append(- mini_points_new_x[-1] / periods[-1] * 2 * np.pi)

    #plt.scatter(mini_points_new_x, mini_points_new_y, c = "blue")

    #plt.scatter(maxi_points_x, maxi_points_y, c = "orange")
    
    #plt.scatter(left_points_x, left_points_y, c = "green")

    #plt.scatter(right_points_x, right_points_y, c = "red")

    #for val in mini_points_new_x:

        #plt.axvline(val, linestyle = "dotted", c = "gray")

    #for val in maxi_points_x:

        #plt.axvline(val, linestyle = "dotted", c = "gray")

    for i in range(len(left_points_y) - 1): 

        x_range = np.arange(mini_points_new_x[i], maxi_points_x[i], 0.01) 

        #plt.plot(x_range, [mini_points_new_y[i] for x in x_range], c = "blue")

        #plt.plot(x_range, [maxi_points_y[i] for x in x_range], c = "orange")

        #plt.plot(x_range, [left_points_y[i] for x in x_range], c = "green")

    for i in range(len(right_points_y) - 1): 

        x_range = np.arange(maxi_points_x[i], mini_points_new_x[i + 1], 0.01)

        y_range = [right_points_y[i] for x in x_range]

        #plt.plot(x_range, [mini_points_new_y[i + 1] for x in x_range], c = "blue")

        #plt.plot(x_range, [maxi_points_y[i] for x in x_range], c = "orange")

        #plt.plot(x_range, y_range, c = "red")

    y_vals = [amplitudes[i] * np.cos(2 * np.pi / periods[i] * x_vals[i] + offsets[i]) + midpoints[i] for i in range(len(x_vals))] 
    ampavg = np.mean(amplitudes)
    peravg = np.mean(periods)
    midavg = np.mean(midpoints)
    y_vals2 = [ampavg * np.cos(2 * np.pi / peravg * x_vals[i]) + midavg for i in range(len(x_vals))] 
    
    plt.plot(waves_train)
    #plt.plot(x_vals, y_vals, linestyle = "dashed")
    plt.plot(x_vals, y_vals)
    #plt.plot(x_vals, y_vals2)

    plt.show()

    plt.plot([abs(a) for a in amplitudes])
    plt.show()
    plt.plot(midpoints)
    plt.show()
    plt.plot(periods)
    plt.show()
 
    rmse_test = math.sqrt(mean_squared_error(waves_train, y_vals)) / (max(waves) - min(waves)) 
    print(rmse_test, np.average(periods)) 