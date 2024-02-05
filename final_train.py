import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os  
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.metrics import mean_squared_error
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
 
def print_error(trainY, testY, train_predict, test_predict, title, range_val):  
    train_RMSE = math.sqrt(mean_squared_error(trainY, train_predict)) 
    test_RMSE = math.sqrt(mean_squared_error(testY, test_predict)) 
    print(title, 'Normalizirani RMSE (treniranje): %.6f RMSE' % (train_RMSE / range_val)) 
    print(title, 'Normalizirani RMSE (testiranje): %.6f RMSE' % (test_RMSE / range_val))   
    return train_RMSE, test_RMSE
 
def plot_result(trainY, testY, train_predict, test_predict, title, datetimes, filename):
    actual = np.append(trainY, testY)  
    predictions = np.append(train_predict, test_predict) 
    rows = len(actual)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(range(rows), actual, color = "b") 
    plt.plot(range(rows), predictions, color = "orange") 
    datetimes_new = datetimes[-len(predictions):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 10))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.axvline(x = len(trainY), color = 'r')
    plt.text(len(trainY) / 2, min(min(actual), min(predictions)), "Treniranje") 
    plt.text((len(trainY) + len(trainY) + len(testY)) / 2, min(min(actual), min(predictions)), "Testiranje", color = "r")  
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

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";") 
    datetimes = list(file_data["date"]) 
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)

    for model_name in os.listdir("train_net/" + filename_no_csv + "/models"):

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/predictions/train/" + model_name)

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name) 

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/plots/" + model_name) 

        ws_array = []
        hidden_array = []
        val_RMSE = []

        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
            val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)
            
            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
 
        print(filename_no_csv, model_name, min(val_RMSE), hidden_array[val_RMSE.index(min(val_RMSE))], ws_array[val_RMSE.index(min(val_RMSE))])

        hidden = hidden_array[val_RMSE.index(min(val_RMSE))]
        ws = ws_array[val_RMSE.index(min(val_RMSE))]

        x_wave_heights, y_wave_heights = get_XY(wave_heights, ws, num_props)

        xtrain = np.array(x_wave_heights[:int(np.floor(len(x_wave_heights) * 0.7))]) 
        xtest = np.array(x_wave_heights[int(np.floor(len(x_wave_heights) * 0.7)):])

        ytrain = np.array(y_wave_heights[:int(np.floor(len(y_wave_heights) * 0.7))]) 
        ytest = np.array(y_wave_heights[int(np.floor(len(y_wave_heights) * 0.7)):])  

        if model_name == "RNN": 
            demo_model = create_RNN(hidden, num_props, (ws, num_props))  

        if model_name == "LSTM":
            demo_model = create_LSTM(hidden, num_props, (ws, num_props))  

        if model_name == "GRU":
            demo_model = create_GRU(hidden, num_props, (ws, num_props))  
        
        demo_model.save("final_train_net/" + filename_no_csv + "/models/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".h5") 
        history_model = demo_model.fit(xtrain, ytrain, verbose = 1)  

        predict_train = demo_model.predict(xtrain)  
        predict_test = demo_model.predict(xtest) 

        print_predictions(ytrain, predict_train, "final_train_net/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_train.csv") 
        print_predictions(ytest, predict_test, "final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_test.csv") 
  
        plot_result(ytrain, ytest, predict_train, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "final_train_net/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".png")
        train_RMSE, test_RMSE = print_error(ytrain, ytest, predict_train, predict_test, "Visina površine mora (m)", max(wave_heights) - min(wave_heights))