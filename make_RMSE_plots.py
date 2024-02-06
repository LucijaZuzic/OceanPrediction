from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math  
import pandas as pd
import os

def print_error(trainY, valY, testY, train_predict, val_predict, test_predict, title, range_val):  
    train_RMSE = math.sqrt(mean_squared_error(trainY, train_predict))
    val_RMSE = math.sqrt(mean_squared_error(valY, val_predict)) 
    test_RMSE = math.sqrt(mean_squared_error(testY, test_predict)) 
    print(title, 'Normalizirani RMSE (treniranje): %.6f RMSE' % (train_RMSE / range_val))
    print(title, 'Normalizirani RMSE (validacija): %.6f RMSE' % (val_RMSE / range_val))
    print(title, 'Normalizirani RMSE (testiranje): %.6f RMSE' % (test_RMSE / range_val))   
    return train_RMSE / range_val, val_RMSE / range_val, test_RMSE / range_val

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"]) 
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)

    train_RMSE, val_RMSE, test_RMSE = dict(), dict(), dict()

    ws_range = set()

    hidden_range = set()
 
    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):  
            
        if not os.path.isdir("train_net/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/plots/" + model_name) 
        
        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/test/" + model_name): 
            
            test_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename, index_col = False, sep = ";") 
            
            ytest = list(test_data["actual"])
            predict_test = list(test_data["predicted"]) 

            ws = filename.replace(".csv", "").split("_")[-4] 
            ws_range.add(ws)
            hidden = filename.replace(".csv", "").split("_")[-2] 
            hidden_range.add(hidden)
        
            train_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename.replace("test", "train"), index_col = False, sep = ";") 
            
            ytrain = list(train_data["actual"])
            predict_train = list(train_data["predicted"])
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename.replace("test", "validate"), index_col = False, sep = ";") 
            
            yval = list(val_data["actual"])
            predict_val = list(val_data["predicted"])

            train_RMSE[((model_name, ws, hidden))], val_RMSE[((model_name, ws, hidden))], test_RMSE[((model_name, ws, hidden))] = print_error(ytrain, yval, ytest, predict_train, predict_val, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", max(wave_heights) - min(wave_heights))

    ws_range = sorted(list(ws_range))
    hidden_range = sorted(list(hidden_range))

    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):  
        
        if not os.path.isdir("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name) 
        
        for ws in ws_range: 
         
            train_RMSE_filtered_ws = []
            val_RMSE_filtered_ws = []
            test_RMSE_filtered_ws = [] 

            for hidden in hidden_range: 
                train_RMSE_filtered_ws.append(train_RMSE[(model_name, ws, hidden)]) 
                val_RMSE_filtered_ws.append(val_RMSE[(model_name, ws, hidden)]) 
                test_RMSE_filtered_ws.append(test_RMSE[(model_name, ws, hidden)]) 

            plt.figure(figsize = (15, 6), dpi = 80)
            plt.title("Normalizirani RMSE za " + model_name + " model (veličina prozora " + str(ws) + ")")
            plt.xlabel("Broj skrivenih jedinica")
            plt.ylabel("Normalizirani RMSE")
            plt.plot(hidden_range, train_RMSE_filtered_ws, label = "Treniranje")
            plt.plot(hidden_range, val_RMSE_filtered_ws, label = "Validacija")
            plt.plot(hidden_range, test_RMSE_filtered_ws, label = "Testiranje")
            plt.legend()
            plt.savefig("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_RMSE.png", bbox_inches = "tight")
            plt.close()

        for hidden in hidden_range: 
         
            train_RMSE_filtered_hidden = []
            val_RMSE_filtered_hidden = []
            test_RMSE_filtered_hidden = [] 

            for ws in ws_range:  
                train_RMSE_filtered_hidden.append(train_RMSE[(model_name, ws, hidden)]) 
                val_RMSE_filtered_hidden.append(val_RMSE[(model_name, ws, hidden)]) 
                test_RMSE_filtered_hidden.append(test_RMSE[(model_name, ws, hidden)]) 

            plt.figure(figsize = (15, 6), dpi = 80)
            plt.title("Normalizirani RMSE za " + model_name + " model (" + str(hidden) + " skrivenih slojeva)")
            plt.xlabel("Veličina prozora")
            plt.ylabel("Normalizirani RMSE")
            plt.plot(ws_range, train_RMSE_filtered_hidden, label = "Treniranje")
            plt.plot(ws_range, val_RMSE_filtered_hidden, label = "Validacija")
            plt.plot(ws_range, test_RMSE_filtered_hidden, label = "Testiranje")
            plt.legend()
            plt.savefig("train_net/" + filename_no_csv + "/RMSE_plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_hidden_" + str(hidden) + "_RMSE.png", bbox_inches = "tight")
            plt.close()