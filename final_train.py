import pandas as pd
import numpy as np
import os  
from sklearn.metrics import mean_squared_error
import math 
from utilities import get_XY, create_GRU, create_LSTM, create_RNN, print_predictions

num_props = 1

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)

    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/predictions/train/" + model_name)

        if not os.path.isdir("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name)

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