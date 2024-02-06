import pandas as pd
import numpy as np
import os  
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import math 
from utilities import get_XY, create_GRU, create_LSTM, create_RNN, print_predictions

num_props = 1
   
def print_extrapolated(dates, predicted, name_file):
    
    strpr = "dates;predicted\n"

    for ix in range(len(dates)):

        strpr += str(dates[ix]) + ";" + str(predicted[ix]) + "\n"

    file_processed = open(name_file, "w")
    file_processed.write(strpr.replace("[", "").replace("]", ""))
    file_processed.close() 

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";") 
    datetimes = list(file_data["date"]) 
    datetimes_value = [datetime.strptime(val, "%d.%m.%Y.") for val in datetimes]
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name) 

        if not os.path.isdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name)   

        ws_array = []
        hidden_array = []
        val_RMSE = []

        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
            val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)
            
            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
 
        print(filename_no_csv, model_name, min(val_RMSE), hidden_array[val_RMSE.index(min(val_RMSE))], ws_array[val_RMSE.index(min(val_RMSE))])
 
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
             
            print_extrapolated(date_strings, predictions_array, "extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".csv")