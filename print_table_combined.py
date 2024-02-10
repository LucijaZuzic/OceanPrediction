import numpy as np
import pandas as pd
import os   
from sklearn.metrics import mean_squared_error
import math

num_props = 1    

dict_for_table = dict()
dict_for_table93 = dict()
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

        ws_array93 = []
        hidden_array93 = []
        val_RMSE93 = []
        filename_array93 = []
         
        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
            
            one_ws = int(filename.replace(".csv", "").split("_")[-4])

            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(one_ws)
            filename_array.append(filename)
            val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)

            if one_ws != 93:
                continue

            hidden_array93.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array93.append(one_ws)
            filename_array93.append(filename)
            val_RMSE93.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)
 
        hidden = hidden_array[val_RMSE.index(min(val_RMSE))]
        ws = ws_array[val_RMSE.index(min(val_RMSE))]
        new_filename = filename_array[val_RMSE.index(min(val_RMSE))]
 
        test_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + new_filename.replace("validate", "test"), index_col = False, sep = ";") 
        test_RMSE = math.sqrt(mean_squared_error(list(test_data["actual"]), list(test_data["predicted"]))) / range_val
           
        dict_for_table[(filename_no_csv, model_name)] = (ws, hidden, np.round(min(val_RMSE) * 1000, 3), np.round(test_RMSE * 1000, 3))

        hidden93 = hidden_array93[val_RMSE93.index(min(val_RMSE93))]
        ws93 = ws_array93[val_RMSE93.index(min(val_RMSE93))]
        new_filename93 = filename_array93[val_RMSE93.index(min(val_RMSE93))]
 
        test_data93 = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + new_filename93.replace("validate", "test"), index_col = False, sep = ";") 
        test_RMSE93 = math.sqrt(mean_squared_error(list(test_data93["actual"]), list(test_data93["predicted"]))) / range_val
           
        dict_for_table93[(filename_no_csv, model_name)] = (ws93, hidden93, np.round(min(val_RMSE93) * 1000, 3), np.round(test_RMSE93 * 1000, 3)) 

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
        ws93, hidden93, val_RMSE93, test_RMSE93 = dict_for_table93[entry]
        print(entry[0].replace("_", " & ") + " & " + str(ws) + " & " + str(hidden) + " & " + str(val_RMSE) + " & " + str(test_RMSE) + " & " + str(hidden93) + " & " + str(val_RMSE93) + " & " + str(test_RMSE93) + " \\\\ \\hline")
         
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
        ws93, hidden93, val_RMSE93, test_RMSE93 = dict_for_table93[entry] 
        print(entry[0].replace("_", " & ") + " & " + str(ws) + " & " + str(hidden) + " & " + str(val_RMSE) + " & " + str(test_RMSE) + " \\\\ \\hline")
         
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
        ws93, hidden93, val_RMSE93, test_RMSE93 = dict_for_table93[entry] 
        print(entry[0].replace("_", " & ") + " & " + str(hidden93) + " & " + str(val_RMSE93) + " & " + str(test_RMSE93) + " \\\\ \\hline")