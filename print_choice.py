import matplotlib.pyplot as plt
import pandas as pd
import os   
from sklearn.metrics import mean_squared_error
import math

num_props = 1   

best_models = []

model_names = set()

best_hidden_for_model_ws = dict()

best_ws_for_model_hidden = dict()

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):

        ws_array = []
        hidden_array = []
        val_RMSE = []

        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/validate/" + model_name): 
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename, index_col = False, sep = ";")  
            val_RMSE.append(math.sqrt(mean_squared_error(list(val_data["actual"]), list(val_data["predicted"]))) / range_val)
            
            hidden_array.append(int(filename.replace(".csv", "").split("_")[-2]))
            ws_array.append(int(filename.replace(".csv", "").split("_")[-4]))
 
        print("validation", filename_no_csv, model_name, min(val_RMSE), hidden_array[val_RMSE.index(min(val_RMSE))], ws_array[val_RMSE.index(min(val_RMSE))])

        best_models.append((model_name, ws_array[val_RMSE.index(min(val_RMSE))], hidden_array[val_RMSE.index(min(val_RMSE))]))

        model_names.add(model_name)

        for ws in sorted(list(set(ws_array))):

            filtered_hidden = []
            filtered_RMSE = []

            for ix_hidden in range(len(ws_array)):

                if ws_array[ix_hidden] == ws:
                    filtered_hidden.append(hidden_array[ix_hidden])
                    filtered_RMSE.append(val_RMSE[ix_hidden])

            if (model_name, ws) not in best_hidden_for_model_ws:
                best_hidden_for_model_ws[(model_name, ws)] = []

            best_hidden_for_model_ws[(model_name, ws)].append(filtered_hidden[filtered_RMSE.index(min(filtered_RMSE))])
            
        for hidden in sorted(list(set(hidden_array))):

            filtered_ws = []
            filtered_RMSE = []

            for ix_ws in range(len(hidden_array)):

                if hidden_array[ix_ws] == hidden:
                    filtered_ws.append(ws_array[ix_ws])
                    filtered_RMSE.append(val_RMSE[ix_ws])

            if (model_name, hidden) not in best_ws_for_model_hidden:
                best_ws_for_model_hidden[(model_name, hidden)] = []

            best_ws_for_model_hidden[(model_name, hidden)].append(filtered_ws[filtered_RMSE.index(min(filtered_RMSE))])

if not os.path.isdir("chosen_sizes"):
    os.makedirs("chosen_sizes")

for model_name in model_names:

    filtered_ws = []
    filtered_hidden = []

    for val in best_models:

        if val[0] == model_name: 
            filtered_ws.append(val[1])
            filtered_hidden.append(val[2])

    count_ws = {val: filtered_ws.count(val) for val in sorted(list(set(filtered_ws)))}
    count_hidden = {val: filtered_hidden.count(val) for val in sorted(list(set(filtered_hidden)))}

    print(model_name)
    print(count_ws)
    print(count_hidden)

    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})

    plt.subplot(1, 2, 1) 
    plt.title("Učestalost odabrane veličine\nprozora za " + model_name + " model")
    plt.pie(list(count_ws.values()), labels = list(count_ws.keys()), autopct = '%1.2f%%')  

    plt.subplot(1, 2, 2) 
    plt.title("Učestalost odabranog broja skrivenih\nslojeva za " + model_name + " model")
    plt.pie(list(count_hidden.values()), labels = list(count_hidden.keys()), autopct = '%1.2f%%') 

    plt.savefig("chosen_sizes/ws_hidden_" + model_name + ".png", bbox_inches = "tight")
    plt.close()

    filtered_best_hidden_for_model_ws = dict()
    filtered_best_ws_for_model_hidden = dict()

    for val in best_hidden_for_model_ws:
        if val[0] == model_name: 
            filtered_best_hidden_for_model_ws[val[1]] = {hidden: best_hidden_for_model_ws[val].count(hidden) for hidden in sorted(list(set(best_hidden_for_model_ws[val])))}

    for val in best_ws_for_model_hidden:
        if val[0] == model_name: 
            filtered_best_ws_for_model_hidden[val[1]] = {ws: best_ws_for_model_hidden[val].count(ws) for ws in sorted(list(set(best_ws_for_model_hidden[val])))}

    for ws in filtered_best_hidden_for_model_ws:

        print(ws, filtered_best_hidden_for_model_ws[ws])
 
        plt.figure(figsize = (15, 6), dpi = 80)
        plt.rcParams.update({'font.size': 22})
        plt.title("Učestalost odabranog broja skrivenih\nslojeva za " + model_name + " model\nza zadanu veličinu prozora " + str(ws))
        plt.pie(list(filtered_best_hidden_for_model_ws[ws].values()), labels = list(filtered_best_hidden_for_model_ws[ws].keys()), autopct = '%1.2f%%') 

        plt.savefig("chosen_sizes/ws_" + str(ws) + "_" + model_name + ".png", bbox_inches = "tight")
        plt.close()

    for hidden in filtered_best_ws_for_model_hidden:

        print(hidden, filtered_best_ws_for_model_hidden[hidden])
 
        plt.figure(figsize = (15, 6), dpi = 80)
        plt.rcParams.update({'font.size': 22})
        plt.title("Učestalost odabrane veličine\nprozora za " + model_name + " model\nza zadani broj skrivenih slojeva " + str(hidden))
        plt.pie(list(filtered_best_ws_for_model_hidden[hidden].values()), labels = list(filtered_best_ws_for_model_hidden[hidden].keys()), autopct = '%1.2f%%') 

        plt.savefig("chosen_sizes/hidden_" + str(hidden) + "_" + model_name + ".png", bbox_inches = "tight")
        plt.close()