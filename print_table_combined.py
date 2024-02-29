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
        filename_array = []
 
        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
             
            is_a_nan = False
            for val in val_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

            if is_a_nan:
                val_RMSE.append(1000000)
            else: 
                val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)

            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
            filename_array.append(filename)
  
        hidden = hidden_array[val_RMSE.index(min(val_RMSE))]
        ws = ws_array[val_RMSE.index(min(val_RMSE))]
        new_filename = filename_array[val_RMSE.index(min(val_RMSE))]
 
        test_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + new_filename.replace("validate", "test"), index_col = False, sep = ";") 
        
        is_a_nan = False
        for val in test_data["predicted"]:
            if str(val) == 'nan':
                is_a_nan = True
                break

        if is_a_nan:
            test_RMSE = 1000000
        else: 
            test_RMSE = math.sqrt(mean_squared_error(list(test_data["actual"]), list(test_data["predicted"]))) / range_val
           
        dict_for_table[(filename_no_csv, model_name)] = (ws, hidden, np.round(min(val_RMSE) * 100, 3), np.round(test_RMSE * 100, 3))
  
        model_names.add(model_name)
  
list_of_locs = []

for filename_no_csv in os.listdir("train_net"):  
  
    long = float(filename_no_csv.split("_")[0])
    lat = float(filename_no_csv.split("_")[1])
 
    list_of_locs.append((long, lat))

list_of_locs.sort()
 
for model_name in model_names:
    val_RMSE_arr = []
    test_RMSE_arr = []
    diff_arr = []
    abs_diff_arr = []
    entries = []
    print(model_name)
    for loc in list_of_locs:
        long, lat = loc
        entry = (str(long) + "_" + str(lat), model_name)
        ws, hidden, val_RMSE, test_RMSE = dict_for_table[entry] 
        print(entry[0].replace("_", " & ") + " & " + str(ws) + " & " + str(hidden) + " & " + str(val_RMSE) + " & " + str(test_RMSE) + " \\\\ \\hline")