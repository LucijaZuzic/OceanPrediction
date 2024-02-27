import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import os

def plot_predictions(ws, previous_predicted, previous_waves, previous_dates, predicted, dates_predicted, title, filename):
    datetimes = np.append(previous_dates, dates_predicted)
    all_predict = np.append(previous_predicted, predicted)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(previous_waves)), previous_waves, color = "b") 
    plt.plot(range(len(all_predict)), all_predict, color = "orange") 
    datetimes_new = datetimes[-len(all_predict):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.axvline(x = len(previous_waves), color = 'r')
    plt.text(len(all_predict) / 4, min(min(all_predict), min(previous_waves)), "Poznate vrijednosti")
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTreniranje i nepoznate vrijednosti")
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

    new_len = len(all_predict) - len(previous_waves)

    datetimes = datetimes[-new_len-ws:]
    all_predict = all_predict[-new_len-ws:]
    previous_waves = previous_waves[-ws:]

    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(previous_waves)), previous_waves, color = "b") 
    plt.plot(range(len(all_predict)), all_predict, color = "orange") 
    datetimes_new = datetimes[-len(all_predict):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.axvline(x = len(previous_waves), color = 'r')
    plt.text(len(all_predict) / 4, min(min(all_predict), min(previous_waves)), "Poznate vrijednosti")
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTreniranje i nepoznate vrijednosti")
    plt.savefig(filename.replace(".png", "_window_only.png"), bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("extrapolate"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"])   
 
    for model_name in os.listdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate"):  
            
        if not os.path.isdir("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name)  
        
        for filename in os.listdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name): 

            if "longer" in filename:
                continue
            
            extrapolate_data = pd.read_csv("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename, index_col = False, sep = ";") 
             
            dates_strings = list(extrapolate_data["dates"])
            predict_extrapolate = list(extrapolate_data["predicted"])

            extrapolate_data_longer = pd.read_csv("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename.replace(".csv", "_longer.csv"), index_col = False, sep = ";") 
             
            dates_strings_longer = list(extrapolate_data_longer["dates"])
            predict_extrapolate_longer = list(extrapolate_data_longer["predicted"])

            ws = int(filename.replace(".csv", "").split("_")[-4])
            hidden = filename.replace(".csv", "").split("_")[-2] 

            train_data = pd.read_csv("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename.replace("extrapolate", "train"), index_col = False, sep = ";") 
             
            ytrain = list(train_data["actual"])
            predict_train = list(train_data["predicted"])

            plot_predictions(ws, predict_train, ytrain, datetimes, predict_extrapolate, dates_strings, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate_past.png")
            plot_predictions(ws, predict_train, ytrain, datetimes, predict_extrapolate_longer, dates_strings_longer, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate_past_longer.png")