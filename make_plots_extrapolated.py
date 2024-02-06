import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_predictions(predicted, dates_predicted, old_new, title, filename):
    
    datetimes_ix_filter = [i for i in range(0, len(dates_predicted), int(len(dates_predicted) // 10))]
    datetimes_filter = [dates_predicted[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(predicted) 
    plt.title(title)
    plt.axvline(old_new, color = "red")
    if old_new <= 365:
        plt.text((len(predicted) + old_new) / 2, min(predicted), "Ekstrapolacija", color = "red") 
    else:
        plt.text(old_new / 2, min(predicted), "Ekstrapolacija", color = "red") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)") 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("extrapolate"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"]) 
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate"):  
            
        if not os.path.isdir("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name)  

        for filename in os.listdir("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name): 
            
            extrapolate_data = pd.read_csv("extrapolate/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename, index_col = False, sep = ";") 
             
            dates_strings = list(extrapolate_data["dates"])
            predict_extrapolate = list(extrapolate_data["predicted"])

            ws = int(filename.replace(".csv", "").split("_")[-3])
            hidden = filename.replace(".csv", "").split("_")[-1] 
                
            plot_predictions(predict_extrapolate, dates_strings, ws, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate.png")