import numpy as np
import pandas as pd
import os   
from sklearn.metrics import mean_squared_error
import math

num_props = 1    

dict_for_table = dict() 
model_names = set()
ws_by_model = dict()
hidden_by_model = dict()

val_RMSE_file = dict()
test_RMSE_file = dict()

for filename_no_csv in os.listdir("train_net"): 

    val_RMSE_file[filename_no_csv] = dict() 
    test_RMSE_file[filename_no_csv] = dict() 
    
    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):
            
            ws_by_model[model_name] = dict() 
            hidden_by_model[model_name] = dict()  

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

            val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)
            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
            filename_array.append(filename)
  
        hidden = hidden_array[val_RMSE.index(min(val_RMSE))]
        ws = ws_array[val_RMSE.index(min(val_RMSE))]
        new_filename = filename_array[val_RMSE.index(min(val_RMSE))]
 
        test_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + new_filename.replace("validate", "test"), index_col = False, sep = ";") 
        test_RMSE = math.sqrt(mean_squared_error(list(test_data["actual"]), list(test_data["predicted"]))) / range_val
           
        dict_for_table[(filename_no_csv, model_name)] = (ws, hidden, np.round(min(val_RMSE) * 1000, 3), np.round(test_RMSE * 1000, 3)) 
   
        model_names.add(model_name)

        if ws not in ws_by_model[model_name]:
            ws_by_model[model_name][ws] = 0

        ws_by_model[model_name][ws] += 1

        if hidden not in hidden_by_model[model_name]:
            hidden_by_model[model_name][hidden] = 0

        hidden_by_model[model_name][hidden] += 1

        val_RMSE_file[filename_no_csv][model_name] = min(val_RMSE)
        test_RMSE_file[filename_no_csv][model_name] = test_RMSE

max_hidden = {model_name: max(list(hidden_by_model[model_name].values())) for model_name in hidden_by_model}
max_ws = {model_name: max(list(ws_by_model[model_name].values())) for model_name in ws_by_model}

best_for_model_val = dict()
best_for_model_test = dict()

for filename_no_csv in os.listdir("train_net"):  
    ws_vals = dict()
    hidden_vals = dict() 
    for entry in dict_for_table:
        if entry[0] != filename_no_csv:
            continue 
        ws, hidden, val_RMSE, test_RMSE = dict_for_table[entry]
        ws_vals[entry[1]] = ws
        hidden_vals[entry[1]] = hidden
    is_ok = True
    for model_name in ws_vals: 
        if ws_by_model[model_name][ws_vals[model_name]] != max_ws[model_name]:
            is_ok = False
            break
    for model_name in hidden_vals: 
        if hidden_by_model[model_name][hidden_vals[model_name]] != max_hidden[model_name]:
            is_ok = False
            break
    if is_ok:
        print(is_ok, filename_no_csv, ws_vals, hidden_vals)
        for model_name in ws_vals:
            print(model_name, test_RMSE_file[filename_no_csv][model_name], val_RMSE_file[filename_no_csv][model_name])

            if model_name not in best_for_model_test or test_RMSE_file[filename_no_csv][model_name] < best_for_model_test[model_name][1]:
                best_for_model_test[model_name] = (filename_no_csv, test_RMSE_file[filename_no_csv][model_name])
            
            if model_name not in best_for_model_val or val_RMSE_file[filename_no_csv][model_name] < best_for_model_val[model_name][1]:
                best_for_model_val[model_name] = (filename_no_csv, val_RMSE_file[filename_no_csv][model_name])

print(best_for_model_val)
print(best_for_model_test)