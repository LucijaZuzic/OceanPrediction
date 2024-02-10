import pandas as pd
import numpy as np

file_loc = pd.read_csv("location_data/location_long_lat_csv_string.csv", sep = ";")
 
lines_to_print = dict()
list_of_locs = []

for ix in range(len(file_loc["loc"])):
    lines_to_print[file_loc["min_long"][ix]] = str(file_loc["loc"][ix]) + " & " + str(np.round(file_loc["min_long"][ix], 3)) + " & " + str(np.round(file_loc["max_long"][ix], 3)) + " & " + str(np.round(file_loc["min_lat"][ix], 3)) + " & " + str(np.round(file_loc["max_lat"][ix], 3)) + " \\\\ \\hline"
    
    list_of_locs.append(file_loc["min_long"][ix])

list_of_locs.sort()

for loc in list_of_locs:
    print(lines_to_print[loc])