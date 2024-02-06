import numpy as np
import pandas as pd
import os   
from sklearn.metrics import mean_squared_error
import math

num_props = 1    

dict_for_table = dict()
model_names = set()

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):

        ws_array = []
        hidden_array = []
        val_RMSE = []

        for filename in os.listdir("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name): 
            
            test_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename, index_col = False, sep = ";") 
            test_RMSE = math.sqrt(mean_squared_error(list(test_data["actual"]), list(test_data["predicted"]))) / range_val
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename.replace("test", "validate"), index_col = False, sep = ";")  
            val_RMSE = math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val 
            
            hidden = filename.replace(".csv", "").split("_")[-2]
            ws = filename.replace(".csv", "").split("_")[-4]
 
            dict_for_table[(filename_no_csv, model_name)] = (ws, hidden, np.round(val_RMSE * 1000, 3), np.round(test_RMSE * 1000, 3)) 

            model_names.add(model_name)

for model_name in model_names:
    print(model_name)
    for entry in dict_for_table:
        if entry[1] != model_name:
            continue
        ws, hidden, val_RMSE, test_RMSE = dict_for_table[entry]
        print(entry[0].replace("_", " & ") + " & " + ws + " & " + hidden + " & " + str(val_RMSE) + " & " + str(test_RMSE) + " \\\\ \\hline")