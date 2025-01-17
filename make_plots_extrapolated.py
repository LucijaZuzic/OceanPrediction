import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_predictions(predicted, dates_predicted, title, filename):
    
    datetimes_ix_filter = [i for i in range(0, len(dates_predicted), int(len(dates_predicted) // 5))]
    datetimes_filter = [dates_predicted[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(predicted)), predicted, color = "orange") 
    plt.title(title + "\nNepoznate vrijednosti")
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)") 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("extrapolate"):  
 
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
                
            is_a_nan = False

            for val in predict_extrapolate:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

            for val in predict_extrapolate_longer:
                if str(val) == 'nan':
                    is_a_nan = True
                    break
                
            if is_a_nan:
                print(filename_no_csv, ws, hidden, model_name, "error")
            else:
                plot_predictions(predict_extrapolate, dates_strings, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate.png")
                plot_predictions(predict_extrapolate_longer, dates_strings_longer, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", "extrapolate/" + filename_no_csv + "/extrapolate_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_extrapolate_longer.png")