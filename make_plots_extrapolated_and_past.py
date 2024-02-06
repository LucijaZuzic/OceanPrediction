import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import os

def plot_predictions(previous_waves, previous_dates, predicted, dates_predicted, old_new, title, filename):
    all_waves = np.append(previous_waves, predicted) 
    all_dates = np.append(previous_dates, dates_predicted)  
    datetimes_ix_filter = [i for i in range(0, len(all_dates), int(len(all_dates) // 10))]
    datetimes_filter = [all_dates[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(all_waves) 
    plt.title(title)
    plt.axvline(len(previous_waves) + old_new, color = "red")
    plt.text((len(previous_waves) + old_new) * 0.9, min(all_waves),"Ekstrapolacija", color = "red") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)") 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("extrapolate"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"])  
    wave_heights = list(file_data["sla"])  
 
    for model_name in os.listdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate"):  
            
        if not os.path.isdir("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name)  

        for filename in os.listdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name): 
            
            extrapolate_data = pd.read_csv("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename, index_col = False, sep = ";") 
             
            dates_strings = list(extrapolate_data["dates"])
            predict_extrapolate = list(extrapolate_data["predicted"])

            ws = int(filename.replace(".csv", "").split("_")[-3])
            hidden = filename.replace(".csv", "").split("_")[-1] 

            previous_waves = []
            previous_dates = []

            for i in range(len(datetimes)):

                if datetimes[i] == dates_strings[0]:
                    break

                previous_waves.append(wave_heights[i])
                previous_dates.append(datetimes[i])
                
            plot_predictions(previous_waves, previous_dates, predict_extrapolate, dates_strings, ws, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate_past.png")