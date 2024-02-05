import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os  
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import math 

num_props = 1 

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
 
def print_error(trainY, train_predict, title, range_val):  
    train_RMSE = math.sqrt(mean_squared_error(trainY, train_predict)) 
    print(title, 'Normalizirani RMSE (treniranje): %.6f RMSE' % (train_RMSE / range_val)) 
    return train_RMSE
 
def plot_result(trainY, train_predict, title, datetimes, filename):
    rows = len(trainY)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(range(rows), trainY, color = "b") 
    plt.plot(range(rows), train_predict, color = "orange") 
    datetimes_new = datetimes[-len(train_predict):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 10))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Predviđeno'])
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title) 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

def print_predictions(actual, predicted, name_file):
    
    strpr = "actual;predicted\n"

    for ix in range(len(actual)):

        strpr += str(actual[ix]) + ";" + str(predicted[ix]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", ""))
    file_processed.close()

def print_extrapolated(dates, predicted, name_file):
    
    strpr = "dates;predicted\n"

    for ix in range(len(dates)):

        strpr += str(dates[ix]) + ";" + str(predicted[ix]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", ""))
    file_processed.close()
 
def plot_predictions(predicted, dates_predicted, old_new, title, filename):
    
    datetimes_ix_filter = [i for i in range(0, len(dates_predicted), int(len(dates_predicted) // 10))]
    datetimes_filter = [dates_predicted[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(predicted) 
    plt.title(title)
    plt.axvline(old_new, color = "red")
    if old_new <= 365:
        plt.text((len(predicted) + old_new) / 2, min(predicted), "Ekstrapolacija", color = "red") 
    else:
        plt.text(old_new / 2, min(predicted), "Ekstrapolacija", color = "red") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)") 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";") 
    datetimes = list(file_data["date"]) 
    datetimes_value = [datetime.strptime(val, "%d.%m.%Y.") for val in datetimes]
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("train_net/" + filename_no_csv + "/models"):

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name) 

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name) 

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/plots/" + model_name) 
            
        if not os.path.isdir("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name) 

        ws_array = []
        hidden_array = []
        val_RMSE = []

        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
            val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)
            
            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
 
        print(filename_no_csv, model_name, min(val_RMSE), hidden_array[val_RMSE.index(min(val_RMSE))], ws_array[val_RMSE.index(min(val_RMSE))])

        ws = ws_array[val_RMSE.index(min(val_RMSE))]
 
        for ws in sorted(list(set(ws_array))):

            filtered_hidden = []
            filtered_RMSE = []

            for ix_hidden in range(len(ws_array)):

                if ws_array[ix_hidden] == ws:
                    filtered_hidden.append(hidden_array[ix_hidden])
                    filtered_RMSE.append(val_RMSE[ix_hidden])

            hidden = filtered_hidden[filtered_RMSE.index(min(filtered_RMSE))]
        
            xtrain, ytrain = get_XY(wave_heights, ws, num_props)  

            if model_name == "RNN": 
                demo_model = create_RNN(hidden, num_props, (ws, num_props))  

            if model_name == "LSTM":
                demo_model = create_LSTM(hidden, num_props, (ws, num_props))  

            if model_name == "GRU":
                demo_model = create_GRU(hidden, num_props, (ws, num_props))  
            
            demo_model.save("extrapolate/" + filename_no_csv + "/models/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".h5") 
            history_model = demo_model.fit(xtrain, ytrain, verbose = 1)  

            predict_train = demo_model.predict(xtrain)   

            print_predictions(ytrain, predict_train, "extrapolate/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_train.csv") 
    
            plot_result(ytrain, predict_train, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "extrapolate/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".png")
            
            train_RMSE = print_error(ytrain, predict_train, "Visina površine mora (m)", max(wave_heights) - min(wave_heights))
    
            predictions_array = wave_heights[-ws:]  

            date_predicted = datetimes_value[-ws:]

            predictions_array.append(0)

            date_predicted.append(date_predicted[-1] + timedelta(days = 1))

            last_datetime = datetime(year = 2023, month = 8, day = 31)

            ord_pred = 0

            while date_predicted[-1] < last_datetime:

                x_extrapolated, y_extrapolated = get_XY(np.array(predictions_array[-ws-1:]), ws, num_props)  

                preds = demo_model.predict(x_extrapolated)  

                if ord_pred != 0:
                    predictions_array.append(preds[0][0]) 
                    date_predicted.append(date_predicted[-1] + timedelta(days = 1))

                else:
                    predictions_array[-1] = preds[0][0] 
                    ord_pred = 1
            
            date_strings = [datetime.strftime(one_day, "%d.%m.%Y.") for one_day in date_predicted]
            
            plot_predictions(predictions_array, date_strings, ws, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate.png")

            print_extrapolated(date_strings, predictions_array, "extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".csv")