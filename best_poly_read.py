import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error 
  
dict_files = dict()

dict_files_all = dict()
dict_files_train = dict()
dict_files_test = dict()

dict_files_nw_all = dict()
dict_files_nw_train = dict()
dict_files_nw_test = dict()

dict_files_w_all = dict()
dict_files_w_train = dict()
dict_files_w_test = dict()

best_dict_files = dict()

best_dict_files_all = dict()
best_dict_files_train = dict()
best_dict_files_test = dict()

best_dict_files_nw_all = dict()
best_dict_files_nw_train = dict()
best_dict_files_nw_test = dict()

best_dict_files_w_all = dict()
best_dict_files_w_train = dict()
best_dict_files_w_test = dict()

for filename in os.listdir("method_scores"): 
    
    dict_files[filename] = dict()

    dict_files_all[filename] = dict()
    dict_files_train[filename] = dict()
    dict_files_test[filename] = dict()

    dict_files_nw_all[filename] = dict()
    dict_files_nw_train[filename] = dict()
    dict_files_nw_test[filename] = dict()

    dict_files_w_all[filename] = dict()
    dict_files_w_train[filename] = dict()
    dict_files_w_test[filename] = dict()

    file_data = pd.read_csv("method_scores/" + filename, index_col = False, sep = ";")
    
    best = (-1, 1000)
    best_all = (-1, 1000)
    best_train = (-1, 1000)
    best_test = (-1, 1000)

    best_nw_all = (-1, 1000)
    best_nw_train = (-1, 1000)
    best_nw_test = (-1, 1000)

    best_w_all = (-1, 1000)
    best_w_train = (-1, 1000)
    best_w_test = (-1, 1000)

    for val in file_data.columns[1:]:

        dict_files[filename][val] = float(file_data[val][0])

        if float(file_data[val][0]) < best[1]:
            best = (val, float(file_data[val][0]))

        if "train" in val:
            dict_files_train[filename][val] = float(file_data[val][0])
            if float(file_data[val][0]) < best_train[1]:
                best_train = (val, float(file_data[val][0]))
            if "w" in val:
                dict_files_w_train[filename][val] = float(file_data[val][0])
                if float(file_data[val][0]) < best_w_train[1]:
                    best_w_train = (val, float(file_data[val][0]))
            else:
                dict_files_nw_train[filename][val] = float(file_data[val][0])
                if float(file_data[val][0]) < best_nw_train[1]:
                    best_nw_train = (val, float(file_data[val][0]))
                
        if "test" in val:
            dict_files_test[filename][val] = float(file_data[val][0])
            if float(file_data[val][0]) < best_test[1]:
                best_test = (val, float(file_data[val][0]))
            if "w" in val:
                dict_files_w_test[filename][val] = float(file_data[val][0])
                if float(file_data[val][0]) < best_w_test[1]:
                    best_w_test = (val, float(file_data[val][0]))
            else:
                dict_files_nw_test[filename][val] = float(file_data[val][0])
                if float(file_data[val][0]) < best_nw_test[1]:
                    best_nw_test = (val, float(file_data[val][0]))
                
        if "all" in val:
            dict_files_all[filename][val] = float(file_data[val][0])
            if float(file_data[val][0]) < best_all[1]:
                best_all = (val, float(file_data[val][0]))
            if "w" in val:
                dict_files_w_all[filename][val] = float(file_data[val][0])
                if float(file_data[val][0]) < best_w_all[1]:
                    best_w_all = (val, float(file_data[val][0]))
            else:
                dict_files_nw_all[filename][val] = float(file_data[val][0])
                if float(file_data[val][0]) < best_nw_all[1]:
                    best_nw_all = (val, float(file_data[val][0]))

    if best[0] not in best_dict_files:
        best_dict_files[best[0]] = 0
    best_dict_files[best[0]] += 1

    if best_all[0] not in best_dict_files_all:
        best_dict_files_all[best_all[0]] = 0
    best_dict_files_all[best_all[0]] += 1
    
    if best_w_all[0] not in best_dict_files_w_all:
        best_dict_files_w_all[best_w_all[0]] = 0
    best_dict_files_w_all[best_w_all[0]] += 1
    
    if best_nw_all[0] not in best_dict_files_nw_all:
        best_dict_files_nw_all[best_nw_all[0]] = 0
    best_dict_files_nw_all[best_nw_all[0]] += 1
    
    if best_train[0] not in best_dict_files_train:
        best_dict_files_train[best_train[0]] = 0
    best_dict_files_train[best_train[0]] += 1
    
    if best_w_train[0] not in best_dict_files_w_train:
        best_dict_files_w_train[best_w_train[0]] = 0
    best_dict_files_w_train[best_w_train[0]] += 1
    
    if best_nw_train[0] not in best_dict_files_nw_train:
        best_dict_files_nw_train[best_nw_train[0]] = 0
    best_dict_files_nw_train[best_nw_train[0]] += 1

    if best_test[0] not in best_dict_files_test:
        best_dict_files_test[best_test[0]] = 0
    best_dict_files_test[best_test[0]] += 1
    
    if best_w_test[0] not in best_dict_files_w_test:
        best_dict_files_w_test[best_w_test[0]] = 0
    best_dict_files_w_test[best_w_test[0]] += 1
    
    if best_nw_test[0] not in best_dict_files_nw_test:
        best_dict_files_nw_test[best_nw_test[0]] = 0
    best_dict_files_nw_test[best_nw_test[0]] += 1
  
print(best_dict_files_nw_test)
print(best_dict_files_w_test)
print(best_dict_files_test)

print(best_dict_files_nw_train)
print(best_dict_files_w_train)
print(best_dict_files_train)

print(best_dict_files_nw_all)
print(best_dict_files_w_all)
print(best_dict_files_all)

print(best_dict_files)