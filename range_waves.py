import numpy as np
import pandas as pd
import os   
from utilities import load_object
    
location_vehicles = load_object("location_data/location_veh")
vehicle_location = load_object("location_data/vehicle_location")
dict_xval_loc = load_object("location_data/dict_xval_loc")
dict_yval_loc = load_object("location_data/dict_yval_loc")
min_long_loc = load_object("location_data/min_long_loc")
max_long_loc = load_object("location_data/max_long_loc")
min_lat_loc = load_object("location_data/min_lat_loc")
max_lat_loc = load_object("location_data/max_lat_loc")

num_props = 1    

dict_for_table = dict()
model_names = set()

lines_to_print = dict()
list_of_locs = []

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
   
    long = float(filename_no_csv.split("_")[0])
    lat = float(filename_no_csv.split("_")[1])

    str_loc = ""

    for loc in dict_xval_loc: 

        if long >= dict_xval_loc[loc][0] and long <= dict_xval_loc[loc][-1] and lat >= dict_yval_loc[loc][0] and lat <= dict_yval_loc[loc][-1]:
            if str_loc != "":
                str_loc +=  ", "
            str_loc += str(loc)

    lines_to_print[(long, lat)] = filename_no_csv.replace("_", " & ") + " & " + str(max(wave_heights)) + " & " + str(min(wave_heights)) + " & " + str(np.round(range_val, 4)) + " & " + str_loc + " \\\\ \\hline"

    list_of_locs.append((long, lat))

list_of_locs.sort()

for loc in list_of_locs:
    print(lines_to_print[loc])