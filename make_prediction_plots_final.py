from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math 
import numpy as np
import pandas as pd
import os

def print_error(trainY, testY, train_predict, test_predict, title, range_val):  
    train_RMSE = math.sqrt(mean_squared_error(trainY, train_predict)) 
    test_RMSE = math.sqrt(mean_squared_error(testY, test_predict)) 
    print(title, 'Normalizirani RMSE (treniranje): %.6f RMSE' % (train_RMSE / range_val)) 
    print(title, 'Normalizirani RMSE (testiranje): %.6f RMSE' % (test_RMSE / range_val))   
    return train_RMSE / range_val, test_RMSE / range_val
 
def plot_result(trainY, testY, train_predict, test_predict, title, datetimes, filename):
    actual = np.append(trainY, testY)  
    predictions = np.append(train_predict, test_predict) 
    rows = len(actual)
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.plot(range(rows), actual, color = "b") 
    plt.plot(range(rows), predictions, color = "orange") 
    datetimes_new = datetimes[-len(predictions):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 10))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.axvline(x = len(trainY), color = 'r')
    plt.text(len(trainY) / 2, min(min(actual), min(predictions)), "Treniranje") 
    plt.text((len(trainY) + len(trainY) + len(testY)) / 2, min(min(actual), min(predictions)), "Testiranje", color = "r")  
    plt.legend(['Stvarno', 'Predviđeno'])
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title) 
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("final_train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"]) 
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("final_train_net/" + filename_no_csv + "/predictions/test"):  
            
        if not os.path.isdir("final_train_net/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("final_train_net/" + filename_no_csv + "/plots/" + model_name)  
        
        for filename in os.listdir("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name): 
            
            test_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename, index_col = False, sep = ";") 
            
            ytest = list(test_data["actual"])
            predict_test = list(test_data["predicted"]) 

            ws = filename.replace(".csv", "").split("_")[-4] 
            hidden = filename.replace(".csv", "").split("_")[-2] 
        
            train_data = pd.read_csv("final_train_net/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename.replace("test", "train"), index_col = False, sep = ";") 
            
            ytrain = list(train_data["actual"])
            predict_train = list(train_data["predicted"])
                
            plot_result(ytrain, ytest, predict_train, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "final_train_net/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".png")
            train_RMSE, test_RMSE = print_error(ytrain, ytest, predict_train, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", range_val)