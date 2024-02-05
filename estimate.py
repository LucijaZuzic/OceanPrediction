from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
import os 
from utilities import load_object
from scipy import stats
from datetime import datetime
from datetime import timedelta
   
dict_xval_loc = load_object("location_data/dict_xval_loc")
dict_yval_loc = load_object("location_data/dict_yval_loc")

g_const = 9.86
p_const = 1.029
 
start_date = datetime(year = 1993, month = 1, day = 1)

for loc in dict_xval_loc:

    for file in os.listdir("data_raw"):

        file_split = [int(v) for v in file.replace(".nc", "").split("_")]

        if dict_xval_loc[loc][0] >= file_split[0] and dict_xval_loc[loc][0] <= file_split[1] and dict_yval_loc[loc][0] >= file_split[2] and dict_yval_loc[loc][0] <= file_split[3]:
             
            file2read = netcdf.NetCDFFile('data_raw/' + file,'r')
 
            if dict_xval_loc[loc][0] >= file2read.variables["longitude"][0] and dict_xval_loc[loc][0] <= file2read.variables["longitude"][-1] and dict_yval_loc[loc][0] >= file2read.variables["latitude"][0] and dict_yval_loc[loc][0] <= file2read.variables["latitude"][-1]:
                
                for j, lat in enumerate(file2read.variables["latitude"]):

                    for k, long in enumerate(file2read.variables["longitude"]):

                        if long >= dict_xval_loc[loc][0] and long <= dict_xval_loc[loc][-1] and lat >= dict_yval_loc[loc][0] and lat <= dict_yval_loc[loc][-1]:
                            
                            wave_heights = file2read.variables["sla"][:, j, k]   
  
                            if min(wave_heights) != max(wave_heights) and not np.isnan(min(wave_heights)):
                                
                                vgosa = file2read.variables["vgosa"][:, j, k]

                                ugosa = file2read.variables["ugosa"][:, j, k]
                
                                tgosa = [np.sqrt(ugosa[i] ** 2 + vgosa[i] ** 2) for i in range(len(ugosa))]  
                                
                                time_deltas = file2read.variables["time"][0:] - file2read.variables["time"][0]
                                
                                datetimes = [datetime.strftime(start_date + timedelta(days = int(time_delta)), "%d.%m.%Y.") for time_delta in time_deltas]
    
                                plt.plot(wave_heights)  

                                mini_points = []

                                maxi_points = []
  
                                ws = 365

                                for i in range(len(wave_heights)):
                                    
                                    window = list(wave_heights[i: i + ws])

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
  
                                print(len(mini_points_group), len(wave_heights) // ws)
 
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

                                    midpoints.append((mini_points_new_y[0] + wave_heights[0]) / 2)

                                    amplitudes.append((abs(mini_points_new_y[0] - midpoints[-1]) + abs(wave_heights[0] - midpoints[-1])) / 2)
                                        
                                    periods.append(2 * mini_points_new_x[0])

                                    offsets.append(0)

                                maxi_points_x = []

                                maxi_points_y = []

                                left_points_x = []

                                left_points_y = []

                                right_points_x = []

                                right_points_y = []

                                for i in range(len(mini_points_new_x) - 1): 

                                    window = list(wave_heights[mini_points_new_x[i]:mini_points_new_x[i + 1]])

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
   
                                for x_val in range(mini_points_new_x[-1], len(wave_heights)):

                                    x_vals.append(x_val)

                                    midpoints.append((mini_points_new_y[-1] + wave_heights[-1]) / 2)

                                    amplitudes.append(- (abs(mini_points_new_y[-1] - midpoints[-1]) + abs(wave_heights[-1] - midpoints[-1])) / 2)
                                        
                                    periods.append(2 * (len(wave_heights) - mini_points_new_x[-1]))

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

                                #plt.plot(x_vals, y_vals, linestyle = "dashed")
                                plt.plot(x_vals, y_vals)
                                plt.plot(x_vals, y_vals2)
  
                                plt.show()

                                plt.plot([abs(a) for a in amplitudes])
                                plt.show()
                                plt.plot(midpoints)
                                plt.show()
                                plt.plot(periods)
                                plt.show()