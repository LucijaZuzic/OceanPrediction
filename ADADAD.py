from scipy.io import netcdf
import matplotlib.pyplot as plt
import numpy as np
import os 
from utilities import load_object
from scipy import stats
from datetime import datetime
from datetime import timedelta
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, LSTM, ReLU 
from sklearn.metrics import mean_squared_error
   
dict_xval_loc = load_object("location_data/dict_xval_loc")
dict_yval_loc = load_object("location_data/dict_yval_loc")

g_const = 9.86
p_const = 1.029
 
def get_XY(dat, time_steps, num_props): 
    Y_ind = np.arange(time_steps, len(dat), time_steps)  
    for x in Y_ind:
        print(dat[x])
    print(Y_ind)
    Y = dat[Y_ind] 
    rows_x = len(Y) 
    X = dat[range(time_steps * rows_x)]
    X = np.reshape(X, (rows_x, time_steps, num_props))    
    return X, Y

def create_RNN(hidden_units, dense_units, input_shape, act_layer = "linear"):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape = input_shape, activation = act_layer))
    model.add(Dense(units = dense_units, activation = act_layer))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    return model
 
def plot_result(trainY, valY, testY, train_predict, val_predict, test_predict, title):
    actual = np.append(trainY, valY) 
    actual = np.append(actual, testY) 
    predictions = np.append(train_predict, val_predict)
    predictions = np.append(predictions, test_predict)
    rows = len(actual)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(range(rows), actual, color = "b") 
    plt.plot(range(rows), predictions, color = "orange") 
    plt.axvline(x = len(trainY), color = 'r')
    plt.axvline(x = len(trainY) + len(valY), color = 'g')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel(title)
    plt.title('Actual and Predicted Values.\nThe Red Line Separates The Training And Validation Examples.\nThe Green Line Separates The Validation And Testing Examples.\n') 
    plt.show()
    plt.close()
 
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
  
                                peak_x = []

                                peak_y = []

                                abs_peak_y = []

                                ws = 31

                                for i in range(len(wave_heights)):

                                    window = wave_heights[max(0, i - ws):min(len(wave_heights), i + ws)]

                                    if wave_heights[i] == max(window) or wave_heights[i] == min(window):

                                        peak_x.append(i) 

                                        peak_y.append(wave_heights[i])  

                                        abs_peak_y.append(abs(wave_heights[i]))  

                                ptp = [peak_x[i] - peak_x[i - 1] for i in range(1, len(peak_x))]

                                plt.plot(wave_heights)
                                plt.scatter(peak_x, peak_y, c = "red")
                                plt.show() 

                                print(peak_x)

                                print(peak_y)