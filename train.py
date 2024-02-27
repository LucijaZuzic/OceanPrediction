import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from utilities import get_XY, get_X, create_GRU, create_LSTM, create_RNN, print_predictions, print_extrapolated

num_props = 1
 
ws_range = [365, 365 * 2, 365 * 3, 365 * 5, 365 * 7]

hidden_range = range(120, 220, 20)

ws_range = [365 * 2]

hidden_range = [200]

model_list = ["LSTM", "RNN", "GRU"] 

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    wave_heights = list(file_data["sla"])
    
    range_val = max(wave_heights) - min(wave_heights) 

    datetimes = list(file_data["date"])

    datetimes_value = [datetime.strptime(val, "%d.%m.%Y.") for val in datetimes]

    last_datetime = datetime(year = 2023, month = 8, day = 31)

    new_delta = last_datetime - datetimes_value[-1]

    datetimes_value_new = [datetime.strptime(val, "%d.%m.%Y.") for val in datetimes]
    datetimes_value_added = []

    while datetimes_value_new[-1] != last_datetime: 
        datetimes_value_new.append(datetimes_value_new[-1] + timedelta(days = 1))
        datetimes_value_added.append(datetimes_value_new[-1])
    
    datetimes_value_added_string = [datetime.strftime(val, "%d.%m.%Y.") for val in datetimes_value_added]
 
    for model_name in model_list:
  
        if not os.path.isdir("train_net/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/train/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/validate/" + model_name)  

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/test/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/test/" + model_name)
 
        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/extrapolate/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/extrapolate/" + model_name)
        
        for ws in ws_range: 

            x_wave_heights, y_wave_heights = get_XY(wave_heights, ws)
            x_wave_heights_predict = get_X(wave_heights, ws)

            ix_train_val = int(np.floor(len(x_wave_heights_predict) * 0.49))
            ix_val_test = int(np.floor(len(x_wave_heights_predict) * 0.7))

            xtrain = np.array(x_wave_heights[:ix_train_val])
            ytrain = np.array(y_wave_heights[:ix_train_val])
            
            xval = np.array(x_wave_heights[ix_train_val:ix_val_test])
            yval = np.array(y_wave_heights[ix_train_val:ix_val_test])

            xtest = np.array(x_wave_heights[ix_val_test:])
            ytest = np.array(y_wave_heights[ix_val_test:])
  
            ytrain_flat = wave_heights[len(xtrain[0]):(len(xtrain) + 1) * len(xtrain[0])]
            yval_flat = wave_heights[(len(xtrain) + 1) * len(xtrain[0]):(len(xtrain) + len(xval) + 1) * len(xtrain[0])]
            ytest_flat = wave_heights[(len(xtrain) + len(xval) + 1) * len(xtrain[0]):]
            
            for hidden in hidden_range: 
 
                if model_name == "RNN":
                    demo_model = create_RNN(hidden, ws, (ws, num_props)) 

                if model_name == "GRU": 
                    demo_model = create_GRU(hidden, ws, (ws, num_props)) 

                if model_name == "LSTM": 
                    demo_model = create_LSTM(hidden, ws, (ws, num_props)) 

                demo_model.save("train_net/" + filename_no_csv + "/models/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".h5") 
                history_model = demo_model.fit(xtrain, ytrain, verbose = 0, epochs = 70)  

                predict_all = demo_model.predict(x_wave_heights_predict)

                all_predicted_merged = []
                predict_train_flat = []
                predict_val_flat = []
                predict_test_flat = []
                predict_extrapolate_flat = []
                for i in range(len(predict_all)):
                    for j in range(len(predict_all[i])): 
                        all_predicted_merged.append(predict_all[i][j])
                        ix = i * len(predict_all[0]) + j
                        if ix < len(ytrain_flat):
                            predict_train_flat.append(predict_all[i][j])
                            continue
                        if ix < len(ytrain_flat) + len(yval_flat):
                            predict_val_flat.append(predict_all[i][j])
                            continue
                        if ix < len(ytrain_flat) + len(yval_flat) + len(ytest_flat):
                            predict_test_flat.append(predict_all[i][j])
                            continue
                        predict_extrapolate_flat.append(predict_all[i][j])
                                                      
                while len(all_predicted_merged) < len(wave_heights) + new_delta.days - ws:
                    new_predict = demo_model.predict(get_X(all_predicted_merged[-ws:], ws)) 
                    for j in range(len(new_predict[0])): 
                        all_predicted_merged.append(new_predict[0][j])
                        predict_extrapolate_flat.append(new_predict[0][j])
  
                datetimes_value_new_longer = [datetime.strptime(val, "%d.%m.%Y.") for val in datetimes]
                datetimes_value_added_longer = []

                for ix_ex in range(len(predict_extrapolate_flat)): 
                    datetimes_value_new_longer.append(datetimes_value_new_longer[-1] + timedelta(days = 1))
                    datetimes_value_added_longer.append(datetimes_value_new_longer[-1])
                
                datetimes_value_added_string_longer = [datetime.strftime(val, "%d.%m.%Y.") for val in datetimes_value_added_longer]
 
                print_predictions(ytrain_flat, predict_train_flat, "train_net/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_train.csv") 
                print_predictions(yval_flat, predict_val_flat, "train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_validate.csv") 
                print_predictions(ytest_flat, predict_test_flat, "train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_test.csv")
                print_extrapolated(datetimes_value_added_string, predict_extrapolate_flat, "train_net/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate.csv")
                print_extrapolated(datetimes_value_added_string_longer, predict_extrapolate_flat, "train_net/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate_longer.csv")

    break