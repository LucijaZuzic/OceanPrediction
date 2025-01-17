import numpy as np
import pandas as pd
import os   
from sklearn.metrics import mean_squared_error
import math

num_props = 1    

dict_for_table = dict()
model_names = set()

filename_no_csv = "18.125_42.375"

file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
wave_heights = list(file_data["sla"]) 
range_val = max(wave_heights) - min(wave_heights)

for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):
 
    hidden_array = []
    ws_array = []
    val_RMSE = []
    test_RMSE = [] 
    
    for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
        
        val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
        
        one_ws = int(filename.replace(".csv", "").split("_")[-4])

        if one_ws > 2 * 365:
            continue
 
        ws_array.append(one_ws) 
        hidden_array.append(int(filename.replace(".csv", "").split("_")[-2])) 
        
        is_a_nan = False
        for val in val_data["predicted"]:
            if str(val) == 'nan':
                is_a_nan = True
                break

        if is_a_nan:
            val_RMSE.append(1000000)
        else: 
            val_RMSE.append(np.round(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val * 100, 3))
  
        if os.path.isfile("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename.replace("validate", "test")):
            test_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename.replace("validate", "test"), index_col = False, sep = ";")  
            
            is_a_nan = False
            for val in test_data["predicted"]:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

            if is_a_nan:
                test_RMSE.append(1000000)
            else: 
                test_RMSE.append(np.round(math.sqrt(mean_squared_error(list(test_data["actual"]), list(test_data["predicted"]))) / range_val * 100, 3))
        else:
            test_RMSE.append(0) 
 
    sorted_hidden = sorted(hidden_array)

    ix_order = [hidden_array.index(hval) for hval in sorted_hidden]

    str_line = "Model"
    for ix in ix_order:
        str_line += " & " + str(hidden_array[ix])
    print(str_line + " \\\\ \\hline")

    str_line = model_name
    for ix in ix_order:
        str_line += " & " + str(ws_array[ix])
    print(str_line + " \\\\ \\hline")

    str_line = model_name
    for ix in ix_order:
        str_line += " & " + str(val_RMSE[ix])
    print(str_line + " \\\\ \\hline")

    str_line = model_name
    for ix in ix_order:
        str_line += " & " + str(test_RMSE[ix])
    print(str_line + " \\\\ \\hline")