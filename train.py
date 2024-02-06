import pandas as pd
import numpy as np
import os
from utilities import get_XY, create_GRU, create_LSTM, create_RNN, print_predictions

num_props = 1
 
ws_range = [1, 7, 31, 2 * 31, 3 * 31]

hidden_range = range(20, 120, 20)

model_list = ["LSTM", "RNN", "GRU"] 

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    wave_heights = list(file_data["sla"]) 

    for model_name in model_list:
  
        if not os.path.isdir("train_net/" + filename_no_csv + "/models/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/models/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/train/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/train/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/test/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/test/" + model_name)

        if not os.path.isdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/predictions/validate/" + model_name)  
  
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