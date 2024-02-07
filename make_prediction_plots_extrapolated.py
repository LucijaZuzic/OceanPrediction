from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math  
import pandas as pd
import os

def print_error(trainY, train_predict, title, range_val):  
    train_RMSE = math.sqrt(mean_squared_error(trainY, train_predict)) 
    print(title, 'Normalizirani RMSE (treniranje): %.6f RMSE' % (train_RMSE / range_val)) 
    return train_RMSE / range_val
 
def plot_result(trainY, train_predict, title, datetimes, filename):
    rows = len(trainY)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(rows), trainY, color = "b") 
    plt.plot(range(rows), train_predict, color = "orange") 
    datetimes_new = datetimes[-len(train_predict):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Predviđeno'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title) 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("extrapolate"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"]) 
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("extrapolate/" + filename_no_csv + "/predictions/train"):  
            
        if not os.path.isdir("extrapolate/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("extrapolate/" + filename_no_csv + "/plots/" + model_name) 
        
        for filename in os.listdir("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name): 
            
            train_data = pd.read_csv("extrapolate/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename, index_col = False, sep = ";") 
             
            ytrain = list(train_data["actual"])
            predict_train = list(train_data["predicted"])

            ws = filename.replace(".csv", "").split("_")[-4] 
            hidden = filename.replace(".csv", "").split("_")[-2] 
                
            plot_result(ytrain, predict_train, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "extrapolate/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".png")
            train_RMSE = print_error(ytrain, predict_train, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", range_val)