import pandas as pd
import numpy as np
import os    
import matplotlib.pyplot as plt 
import pickle
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
location_vehicles = load_object("location_data/location_veh")
vehicle_location = load_object("location_data/vehicle_location")
dict_xval_loc = load_object("location_data/dict_xval_loc")
dict_yval_loc = load_object("location_data/dict_yval_loc")
min_long_loc = load_object("location_data/min_long_loc")
max_long_loc = load_object("location_data/max_long_loc")
min_lat_loc = load_object("location_data/min_lat_loc")
max_lat_loc = load_object("location_data/max_lat_loc")

location_long_lat_csv_string = "loc;min_long;max_long;min_lat;max_lat\n"
   
for loc in dict_xval_loc: 

    location_long_lat_csv_string += str(loc) + ";" + str(min_long_loc[loc]) + ";" + str(max_long_loc[loc]) + ";" + str(min_lat_loc[loc]) + ";" + str(max_lat_loc[loc]) + "\n"
 
    location_point_string = "long;lat\n"

    long_vals = []
    lat_vals = []

    print(loc)

    for filename in os.listdir("processed"):

        file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

        filename_no_csv = filename.replace(".csv", "")
        long = float(filename_no_csv.split("_")[0])
        lat = float(filename_no_csv.split("_")[1])
        
        if long >= dict_xval_loc[loc][0] and long <= dict_xval_loc[loc][-1] and lat >= dict_yval_loc[loc][0] and lat <= dict_yval_loc[loc][-1]:
                            
            long_vals.append(long)
            lat_vals.append(lat)
            location_point_string += str(long) + ";" + str(lat) + "\n"
                
            print(filename_no_csv, long, lat)
    
    if len(long_vals) == 0:
        continue

    step = 0.0001

    long_range = np.arange(min_long_loc[loc], max_long_loc[loc] + step, step)
    lat_range = np.arange(min_lat_loc[loc], max_lat_loc[loc] + step, step)
 
    min_long = [min_long_loc[loc] for lat in lat_range]
    max_long = [max_long_loc[loc] for lat in lat_range]
    min_lat = [min_lat_loc[loc] for lon in long_range]
    max_lat = [max_lat_loc[loc] for lon in long_range]

    xtick_vals = list(set(sorted(long_vals)))
    xtick_vals_range = np.arange(min(xtick_vals), max(xtick_vals) + step, step)
    ytick_vals = list(set(sorted(lat_vals)))
    ytick_vals_range = np.arange(min(ytick_vals), max(ytick_vals) + step, step)
     
    fig = plt.figure(figsize = (6, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.gca().set_aspect('equal') 
    plt.title("Lokacija " + str(loc))
    plt.xlabel("Geografska dužina")
    plt.ylabel("Geografska širina")

    plt.plot(long_range, min_lat, color = "r")
    plt.plot(long_range, max_lat, color = "r")
    plt.plot(min_long, lat_range, color = "r")
    plt.plot(max_long, lat_range, color = "r") 

    for xval in xtick_vals: 
        plt.plot([xval for yval in ytick_vals_range], ytick_vals_range, color = "g")
        
    for yval in ytick_vals: 
        plt.plot(xtick_vals_range, [yval for xval in xtick_vals_range], color = "g") 
 
    plt.xlim(min(min_long_loc[loc], min(xtick_vals)) - 0.01, max(max_long_loc[loc], max(xtick_vals)) + 0.01)
    plt.ylim(min(min_lat_loc[loc], min(ytick_vals)) - 0.01, max(max_lat_loc[loc], max(ytick_vals)) + 0.01)

    plt.scatter(long_vals, lat_vals, color = "b", zorder = 2)

    plt.savefig("location_data/location_point_string_" + str(loc) + ".png", bbox_inches = "tight")
    plt.close()

    location_point_string_file = open("location_data/location_point_string_" + str(loc) + ".csv", "w")
    location_point_string_file.write(location_point_string)
    location_point_string_file.close()
    
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize = (6, 6), dpi = 80) 
    plt.rcParams.update({'font.size': 22})
    
    main_ax = fig.add_subplot(1, 1, 1, projection = proj) 
    main_ax.set_extent([min(min_long_loc[loc], min(xtick_vals)) - 0.01, max(max_long_loc[loc], max(xtick_vals)) + 0.01, min(min_lat_loc[loc], min(ytick_vals)) - 0.01, max(max_lat_loc[loc], max(ytick_vals)) + 0.01], crs = proj)
    main_ax.gridlines(draw_labels = True)
    main_ax.coastlines()

    request = cimgt.OSM()
    main_ax.add_image(request, 12)

    plt.plot(long_range, min_lat, color = "r")
    plt.plot(long_range, max_lat, color = "r")
    plt.plot(min_long, lat_range, color = "r")
    plt.plot(max_long, lat_range, color = "r") 

    for xval in xtick_vals: 
        plt.plot([xval for yval in ytick_vals_range], ytick_vals_range, color = "g")
        
    for yval in ytick_vals: 
        plt.plot(xtick_vals_range, [yval for xval in xtick_vals_range], color = "g") 
  
    plt.scatter(long_vals, lat_vals, color = "b", zorder = 2)

    plt.savefig("location_data/location_point_string_" + str(loc) + "_actual.png", bbox_inches = "tight")
    plt.close()

location_long_lat_csv_string_file = open("location_data/location_long_lat_csv_string.csv", "w")
location_long_lat_csv_string_file.write(location_long_lat_csv_string)
location_long_lat_csv_string_file.close()