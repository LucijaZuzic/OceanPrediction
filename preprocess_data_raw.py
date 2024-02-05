from scipy.io import netcdf 
import numpy as np
import os 
from utilities import load_object 
from datetime import datetime
from datetime import timedelta
     
dict_xval_loc = load_object("location_data/dict_xval_loc")
dict_yval_loc = load_object("location_data/dict_yval_loc")

if not os.path.isdir("processed"):
    os.makedirs("processed")

start_date = datetime(year = 1993, month = 1, day = 1)
   
for loc in dict_xval_loc:

    for file in os.listdir("data_raw"):

        file_split = [int(v) for v in file.replace(".nc", "").split("_")]

        if dict_xval_loc[loc][0] >= file_split[0] and dict_xval_loc[loc][0] <= file_split[1] and dict_yval_loc[loc][0] >= file_split[2] and dict_yval_loc[loc][0] <= file_split[3]:
             
            file2read = netcdf.NetCDFFile('data_raw/' + file,'r')
 
            if dict_xval_loc[loc][0] >= file2read.variables["longitude"][0] and dict_xval_loc[loc][0] <= file2read.variables["longitude"][-1] and dict_yval_loc[loc][0] >= file2read.variables["latitude"][0] and dict_yval_loc[loc][0] <= file2read.variables["latitude"][-1]:
                
                for j, lat in enumerate(file2read.variables["latitude"]):

                    for k, long in enumerate(file2read.variables["longitude"]):

                        if long >= dict_xval_loc[loc][0] and long <= dict_xval_loc[loc][-1] and lat >= dict_yval_loc[loc][0] and lat <= dict_yval_loc[loc][-1]:
                            
                            wave_heights = file2read.variables["sla"][:, j, k]
  
                            if min(wave_heights) != max(wave_heights) and not np.isnan(min(wave_heights)):
                                
                                time_deltas = file2read.variables["time"][0:] - file2read.variables["time"][0]

                                datetimes = [datetime.strftime(start_date + timedelta(days = int(time_delta)), "%d.%m.%Y.") for time_delta in time_deltas]

                                print(datetimes[0], datetimes[-1])
                                print(time_deltas[0], time_deltas[-1])
                                
                                print(long, lat, len(time_deltas), len(wave_heights))

                                strpr = "long;lat;time_delta;date;sla\n"

                                for ix in range(len(wave_heights)):

                                    strpr += str(long) + ";" + str(lat) + ";" + str(time_deltas[ix]) + ";" + str(datetimes[ix]) + ";" + str(wave_heights[ix]) + "\n"

                                file_processed = open("processed/" + str(long) + "_" + str(lat) + ".csv", "w")
                                file_processed.write(strpr)
                                file_processed.close()