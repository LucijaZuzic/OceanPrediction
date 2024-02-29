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
        print(entry[0].replace("_", " & ") + " & " + str(ws) + " & " + str(hidden) + " & " + str(val_RMSE) + " & " + str(test_RMSE) + " & " + str(np.round(test_RMSE - val_RMSE, 3)) + " & " + str(np.round(abs(test_RMSE - val_RMSE), 3)) + " \\\\ \\hline")
        val_RMSE_arr.append(val_RMSE)
        test_RMSE_arr.append(test_RMSE)
        diff_arr.append(np.round(test_RMSE - val_RMSE, 3))
        abs_diff_arr.append(np.round(abs(test_RMSE - val_RMSE), 3))
        entries.append(entry[0])
    print("val_RMSE_arr", "&", np.round(np.quantile(val_RMSE_arr, 0), 3), "&", np.round(np.quantile(val_RMSE_arr, 0.25), 3), "&", np.round(np.quantile(val_RMSE_arr, 0.5), 3), "&", np.round(np.quantile(val_RMSE_arr, 0.75), 3), "&", np.round(np.quantile(val_RMSE_arr, 1), 3), "&", np.round(np.average(val_RMSE_arr), 3), "&", np.round(np.std(val_RMSE_arr), 3), "\\\\ \\hline")
    print("test_RMSE_arr", "&", np.round(np.quantile(test_RMSE_arr, 0), 3), "&", np.round(np.quantile(test_RMSE_arr, 0.25), 3), "&", np.round(np.quantile(test_RMSE_arr, 0.5), 3), "&", np.round(np.quantile(test_RMSE_arr, 0.75), 3), "&", np.round(np.quantile(test_RMSE_arr, 1), 3), "&", np.round(np.average(test_RMSE_arr), 3), "&", np.round(np.std(test_RMSE_arr), 3), "\\\\ \\hline")
    print("diff_arr", "&", np.round(np.quantile(diff_arr, 0), 3), "&", np.round(np.quantile(diff_arr, 0.25), 3), "&", np.round(np.quantile(diff_arr, 0.5), 3), "&", np.round(np.quantile(diff_arr, 0.75), 3), "&", np.round(np.quantile(diff_arr, 1), 3), "&", np.round(np.average(diff_arr), 3), "&", np.round(np.std(diff_arr), 3), "\\\\ \\hline")
    print("abs_diff_arr", "&", np.round(np.quantile(abs_diff_arr, 0), 3), "&", np.round(np.quantile(abs_diff_arr, 0.25), 3), "&", np.round(np.quantile(abs_diff_arr, 0.5), 3), "&", np.round(np.quantile(abs_diff_arr, 0.75), 3), "&", np.round(np.quantile(abs_diff_arr, 1), 3), "&", np.round(np.average(abs_diff_arr), 3), "&", np.round(np.std(abs_diff_arr), 3), "\\\\ \\hline")
    pos_num = np.sum([diff > 0 for diff in diff_arr])
    print(pos_num, "&", np.round(pos_num / len(diff_arr) * 100, 3), "\\\\ \\hline")

    print("val_RMSE_arr max", "&", entries[val_RMSE_arr.index(max(val_RMSE_arr))], "&", val_RMSE_arr[val_RMSE_arr.index(max(val_RMSE_arr))], "&", test_RMSE_arr[val_RMSE_arr.index(max(val_RMSE_arr))], "&", diff_arr[val_RMSE_arr.index(max(val_RMSE_arr))], "&", abs_diff_arr[val_RMSE_arr.index(max(val_RMSE_arr))], "\\\\ \\hline")
    print("test_RMSE_arr max", "&", entries[test_RMSE_arr.index(max(test_RMSE_arr))], "&", val_RMSE_arr[test_RMSE_arr.index(max(test_RMSE_arr))], "&", test_RMSE_arr[test_RMSE_arr.index(max(test_RMSE_arr))], "&", diff_arr[test_RMSE_arr.index(max(test_RMSE_arr))], "&", abs_diff_arr[test_RMSE_arr.index(max(test_RMSE_arr))], "\\\\ \\hline")
    print("diff_arr max", "&", entries[diff_arr.index(max(diff_arr))], "&", val_RMSE_arr[diff_arr.index(max(diff_arr))], "&", test_RMSE_arr[diff_arr.index(max(diff_arr))], "&", diff_arr[diff_arr.index(max(diff_arr))], "&", abs_diff_arr[diff_arr.index(max(diff_arr))], "\\\\ \\hline")
    print("abs_diff_arr max", "&", entries[abs_diff_arr.index(max(abs_diff_arr))], "&", val_RMSE_arr[abs_diff_arr.index(max(abs_diff_arr))], "&", test_RMSE_arr[abs_diff_arr.index(max(abs_diff_arr))], "&", diff_arr[abs_diff_arr.index(max(abs_diff_arr))], "&", max(abs_diff_arr), "\\\\ \\hline")
    
    print("val_RMSE_arr min", "&", entries[val_RMSE_arr.index(min(val_RMSE_arr))], "&", val_RMSE_arr[val_RMSE_arr.index(min(val_RMSE_arr))], "&", test_RMSE_arr[val_RMSE_arr.index(min(val_RMSE_arr))], "&", diff_arr[val_RMSE_arr.index(min(val_RMSE_arr))], "&", abs_diff_arr[val_RMSE_arr.index(min(val_RMSE_arr))], "\\\\ \\hline")
    print("test_RMSE_arr min", "&", entries[test_RMSE_arr.index(min(test_RMSE_arr))], "&", val_RMSE_arr[test_RMSE_arr.index(min(test_RMSE_arr))], "&", test_RMSE_arr[test_RMSE_arr.index(min(test_RMSE_arr))], "&", diff_arr[test_RMSE_arr.index(min(test_RMSE_arr))], "&", abs_diff_arr[test_RMSE_arr.index(min(test_RMSE_arr))], "\\\\ \\hline")
    print("diff_arr min", "&", entries[diff_arr.index(min(diff_arr))], "&", val_RMSE_arr[diff_arr.index(min(diff_arr))], "&", test_RMSE_arr[diff_arr.index(min(diff_arr))], "&", diff_arr[diff_arr.index(min(diff_arr))], "&", abs_diff_arr[diff_arr.index(min(diff_arr))], "\\\\ \\hline")
    print("abs_diff_arr min", "&", entries[abs_diff_arr.index(min(abs_diff_arr))], "&", val_RMSE_arr[abs_diff_arr.index(min(abs_diff_arr))], "&", test_RMSE_arr[abs_diff_arr.index(min(abs_diff_arr))], "&", diff_arr[abs_diff_arr.index(min(abs_diff_arr))], "&", abs_diff_arr[abs_diff_arr.index(min(abs_diff_arr))], "\\\\ \\hline")