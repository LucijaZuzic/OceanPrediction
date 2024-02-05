import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os   
from sklearn.metrics import mean_squared_error
import math 

num_props = 1   

best_models = []

model_names = set()

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
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

        best_models.append((model_name, ws_array[val_RMSE.index(min(val_RMSE))], hidden_array[val_RMSE.index(min(val_RMSE))]))

        model_names.add(model_name)

for model_name in model_names:

    filtered_ws = []
    filtered_hidden = []

    for val in best_models:

        if val[0] == model_name: 
            filtered_ws.append(val[1])
            filtered_hidden.append(val[2])

    count_ws = {val: filtered_ws.count(val) for val in sorted(list(set(filtered_ws)))}
    count_hidden = {val: filtered_hidden.count(val) for val in sorted(list(set(filtered_hidden)))}

    print(model_name)
    print(count_ws)
    print(count_hidden)

    