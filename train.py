import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error
import math 

num_props = 1

min_hidden = 20

max_hidden = 10 * min_hidden

step_hidden = min_hidden

hidden_range = range(min_hidden, max_hidden + step_hidden, step_hidden)

ws_range = [1, 7, 31, 2 * 31, 3 * 31, 4 * 31, 6 * 31, 365, 3 * 365, 7 * 365]
 
ws_range = [1, 7, 31, 2 * 31, 3 * 31]
hidden_range = range(20, 120, 20)

model_list = ["LSTM", "RNN", "GRU"] 

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
 
def print_error(trainY, valY, testY, train_predict, val_predict, test_predict, title, range_val):  
    train_RMSE = math.sqrt(mean_squared_error(trainY, train_predict))
    val_RMSE = math.sqrt(mean_squared_error(valY, val_predict)) 
    test_RMSE = math.sqrt(mean_squared_error(testY, test_predict)) 
    print(title, 'Normalizirani RMSE (treniranje): %.6f RMSE' % (train_RMSE / range_val))
    print(title, 'Normalizirani RMSE (validacija): %.6f RMSE' % (val_RMSE / range_val))
    print(title, 'Normalizirani RMSE (testiranje): %.6f RMSE' % (test_RMSE / range_val))   
    return train_RMSE, val_RMSE, test_RMSE
 
def plot_result(trainY, valY, testY, train_predict, val_predict, test_predict, title, datetimes, filename):
    actual = np.append(trainY, valY) 
    actual = np.append(actual, testY) 
    predictions = np.append(train_predict, val_predict)
    predictions = np.append(predictions, test_predict)
    datetimes_new = datetimes[-len(predictions):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 10))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    rows = len(actual)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(range(rows), actual, color = "b") 
    plt.plot(range(rows), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.axvline(x = len(trainY), color = 'r')
    plt.text(len(trainY) / 2, min(min(actual), min(predictions)), "Treniranje") 
    plt.text((len(trainY) + len(trainY) + len(valY)) / 2, min(min(actual), min(predictions)), "Validacija", color = "r") 
    plt.axvline(x = len(trainY) + len(valY), color = 'g')
    plt.text((len(trainY) + len(valY) + len(trainY) + len(valY) + len(testY)) / 2, min(min(actual), min(predictions)), "Testiranje", color = 'g')
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

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
    
    datetimes = list(file_data["date"])
    wave_heights = list(file_data["sla"])

    train_RMSE = dict()
    val_RMSE = dict()
    test_RMSE = dict() 

    for model_name in model_list:
  
        if not os.path.isdir("train_net/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/train/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/test/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/test/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/validate/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/plots/" + model_name) 

        if not os.path.isdir("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name) 
  
        for ws in ws_range: 

            x_wave_heights, y_wave_heights = get_XY(wave_heights, ws, num_props)

            xtrain = np.array(x_wave_heights[:int(np.floor(len(x_wave_heights) * 0.49))])
            xval = np.array(x_wave_heights[int(np.floor(len(x_wave_heights) * 0.49)):int(np.floor(len(x_wave_heights) * 0.7))])
            xtest = np.array(x_wave_heights[int(np.floor(len(x_wave_heights) * 0.7)):])

            ytrain = np.array(y_wave_heights[:int(np.floor(len(y_wave_heights) * 0.49))])
            yval = np.array(y_wave_heights[int(np.floor(len(y_wave_heights) * 0.49)):int(np.floor(len(y_wave_heights) * 0.7))])
            ytest = np.array(y_wave_heights[int(np.floor(len(y_wave_heights) * 0.7)):]) 

            for hidden in hidden_range: 
                    
                if model_name == "RNN": 
                    demo_model = create_RNN(hidden, num_props, (ws, num_props)) 

                if model_name == "GRU": 
                    demo_model = create_GRU(hidden, num_props, (ws, num_props)) 

                if model_name == "LSTM": 
                    demo_model = create_LSTM(hidden, num_props, (ws, num_props)) 

                demo_model.save("train_net/" + filename_no_csv + "/models/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".h5") 
                history_model = demo_model.fit(xtrain, ytrain, verbose = 1)  

                predict_train = demo_model.predict(xtrain) 
                predict_val = demo_model.predict(xval) 
                predict_test = demo_model.predict(xtest) 

                print_predictions(ytrain, predict_train, "train_net/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_train.csv") 
                print_predictions(yval, predict_val, "train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_validate.csv") 
                print_predictions(ytest, predict_test, "train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_test.csv") 

                plot_result(ytrain, yval, ytest, predict_train, predict_val, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "train_net/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".png")
                train_RMSE[(model_name, ws, hidden)], val_RMSE[(model_name, ws, hidden)], test_RMSE[(model_name, ws, hidden)] = print_error(ytrain, yval, ytest, predict_train, predict_val, predict_test, "Visina površine mora (m)", max(wave_heights) - min(wave_heights))
 
    for model_name in model_list:
        
        for ws in ws_range: 
         
            train_RMSE_filtered_ws = []
            val_RMSE_filtered_ws = []
            test_RMSE_filtered_ws = [] 

            for hidden in hidden_range: 
                train_RMSE_filtered_ws.append(train_RMSE[(model_name, ws, hidden)]) 
                val_RMSE_filtered_ws.append(val_RMSE[(model_name, ws, hidden)]) 
                test_RMSE_filtered_ws.append(test_RMSE[(model_name, ws, hidden)]) 

            plt.figure(figsize = (15, 6), dpi = 80)
            plt.title("Normalizirani RMSE za " + model_name + " model (veličina prozora " + str(ws) + ")")
            plt.xlabel("Broj skrivenih jedinica")
            plt.ylabel("Normalizirani RMSE")
            plt.plot(hidden_range, train_RMSE_filtered_ws, label = "Treniranje")
            plt.plot(hidden_range, val_RMSE_filtered_ws, label = "Validacija")
            plt.plot(hidden_range, test_RMSE_filtered_ws, label = "Testiranje")
            plt.legend()
            plt.savefig("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_RMSE.png", bbox_inches = "tight")
            plt.close()

        for hidden in hidden_range: 
         
            train_RMSE_filtered_hidden = []
            val_RMSE_filtered_hidden = []
            test_RMSE_filtered_hidden = [] 

            for ws in ws_range:  
                train_RMSE_filtered_hidden.append(train_RMSE[(model_name, ws, hidden)]) 
                val_RMSE_filtered_hidden.append(val_RMSE[(model_name, ws, hidden)]) 
                test_RMSE_filtered_hidden.append(test_RMSE[(model_name, ws, hidden)]) 

            plt.figure(figsize = (15, 6), dpi = 80)
            plt.title("Normalizirani RMSE za " + model_name + " model (" + str(hidden) + " skrivenih slojeva)")
            plt.xlabel("Veličina prozora")
            plt.ylabel("Normalizirani RMSE")
            plt.plot(ws_range, train_RMSE_filtered_hidden, label = "Treniranje")
            plt.plot(ws_range, val_RMSE_filtered_hidden, label = "Validacija")
            plt.plot(ws_range, test_RMSE_filtered_hidden, label = "Testiranje")
            plt.legend()
            plt.savefig("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_hidden_" + str(hidden) + "_RMSE.png", bbox_inches = "tight")
            plt.close()