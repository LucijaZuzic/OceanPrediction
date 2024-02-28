from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math 
import numpy as np
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
 
def plot_result(predict_extrapolate, dates_strings, trainY, valY, testY, train_predict, val_predict, test_predict, title, datetimes, filename):
    datetimes = np.append(datetimes, dates_strings)

    actual = np.append(trainY, valY) 
    actual = np.append(actual, testY) 
    predictions = np.append(train_predict, val_predict)
    predictions = np.append(predictions, test_predict)
    predictions = np.append(predictions, predict_extrapolate)
    datetimes_new = datetimes[-len(predictions):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.axvline(x = len(trainY), color = 'r')
    plt.text(len(trainY) / 2, min(min(actual), min(predictions)), "Treniranje") 
    plt.text((len(trainY) + len(trainY) + len(valY)) / 2.15, min(min(actual), min(predictions)), "Validacija", color = "r") 
    plt.axvline(x = len(trainY) + len(valY), color = 'g')
    plt.text((len(trainY) + len(valY) + len(trainY) + len(valY) + len(testY)) / 2, min(min(actual), min(predictions)), "Testiranje", color = 'g')
    plt.axvline(x = len(trainY) + len(valY) + len(testY), color = 'magenta')
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTreniranje, validacija, testiranje i nepoznate vrijednosti") 
    plt.savefig(filename.replace(".png", "_train_validation_test_extrapolate.png"), bbox_inches = "tight")
    plt.close()

    actual = np.append(trainY, valY) 
    actual = np.append(actual, testY) 
    predictions = np.append(train_predict, val_predict)
    predictions = np.append(predictions, test_predict)
    datetimes_new = datetimes[-len(predictions)-len(predict_extrapolate):-len(predict_extrapolate)]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.axvline(x = len(trainY), color = 'r')
    plt.text(len(trainY) / 2, min(min(actual), min(predictions)), "Treniranje") 
    plt.text((len(trainY) + len(trainY) + len(valY)) / 2.15, min(min(actual), min(predictions)), "Validacija", color = "r") 
    plt.axvline(x = len(trainY) + len(valY), color = 'g')
    plt.text((len(trainY) + len(valY) + len(trainY) + len(valY) + len(testY)) / 2, min(min(actual), min(predictions)), "Testiranje", color = 'g')
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTreniranje, validacija i testiranje") 
    plt.savefig(filename.replace(".png", "_train_validation_test.png"), bbox_inches = "tight")
    plt.close()

    actual = np.append(trainY, valY) 
    predictions = np.append(train_predict, val_predict)
    datetimes_new = datetimes[-len(predictions)-len(testY)-len(predict_extrapolate):-len(testY)-len(predict_extrapolate)]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter)
    plt.axvline(x = len(trainY), color = 'r')
    plt.text(len(trainY) / 2, min(min(actual), min(predictions)), "Treniranje") 
    plt.text((len(trainY) + len(trainY) + len(valY)) / 2.15, min(min(actual), min(predictions)), "Validacija", color = "r")
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTreniranje i validacija")  
    plt.savefig(filename.replace(".png", "_train_validation.png"), bbox_inches = "tight")
    plt.close()
    
    actual = np.append(valY, testY) 
    predictions = np.append(val_predict, test_predict)
    datetimes_new = datetimes[-len(predictions)-len(predict_extrapolate):-len(predict_extrapolate)]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.text(len(valY) / 2, min(min(actual), min(predictions)), "Validacija") 
    plt.axvline(x = len(valY), color = 'r')
    plt.text((len(valY) + len(predictions)) / 2, min(min(actual), min(predictions)), "Testiranje", color = 'r')
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nValidacija i testiranje")  
    plt.savefig(filename.replace(".png", "_validation_test.png"), bbox_inches = "tight")
    plt.close()
    
    actual = np.append(valY, testY) 
    predictions = np.append(val_predict, test_predict)
    predictions = np.append(predictions, predict_extrapolate)
    datetimes_new = datetimes[-len(predictions):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.text(len(valY) / 2, min(min(actual), min(predictions)), "Validacija") 
    plt.axvline(x = len(valY), color = 'r')
    plt.axvline(x = len(valY) + len(testY), color = 'g')
    plt.text((len(valY) + len(predictions)) / 2, min(min(actual), min(predictions)), "Testiranje", color = 'r')
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nValidacija, testiranje i nepoznate vrijednosti")  
    plt.savefig(filename.replace(".png", "_validation_test_extrapolate.png"), bbox_inches = "tight")
    plt.close()
    
    actual = testY
    predictions = np.append(test_predict, predict_extrapolate)
    datetimes_new = datetimes[-len(predictions):]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.text(len(testY) / 2, min(min(actual), min(predictions)), "Testiranje") 
    plt.axvline(x = len(testY), color = 'r')
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTestiranje i nepoznate vrijednosti")  
    plt.savefig(filename.replace(".png", "_test_extrapolate.png"), bbox_inches = "tight")
    plt.close()

    actual = trainY
    predictions = train_predict
    datetimes_new = datetimes[-len(predictions)-len(testY)-len(valY)-len(predict_extrapolate):-len(testY)-len(valY)-len(predict_extrapolate)]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTreniranje")  
    plt.savefig(filename.replace(".png", "_train_only.png"), bbox_inches = "tight")
    plt.close()

    actual = valY
    predictions = val_predict
    datetimes_new = datetimes[-len(predictions)-len(testY)-len(predict_extrapolate):-len(testY)-len(predict_extrapolate)]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter] 
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nValidacija")  
    plt.savefig(filename.replace(".png", "_validation_only.png"), bbox_inches = "tight")
    plt.close()

    actual = testY
    predictions = test_predict
    datetimes_new = datetimes[-len(predictions)-len(predict_extrapolate):-len(predict_extrapolate)]
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.xticks(datetimes_ix_filter, datetimes_filter) 
    plt.legend(['Stvarno', 'Modelirano'], loc = "upper left", ncol = 2)
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nTestiranje")  
    plt.savefig(filename.replace(".png", "_test_only.png"), bbox_inches = "tight")
    plt.close()
     
    predictions = predict_extrapolate
    plt.figure(figsize = (15, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22})  
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    datetimes_new = dates_strings
    datetimes_ix_filter = [i for i in range(0, len(datetimes_new), int(len(datetimes_new) // 5))]
    datetimes_filter = [datetimes_new[i] for i in datetimes_ix_filter]
    plt.xticks(datetimes_ix_filter, datetimes_filter)   
    plt.xlabel('Datum')
    plt.ylabel("Visina površine mora (m)")
    plt.title(title + "\nNepoznate vrijednosti")  
    plt.savefig(filename.replace(".png", "_extrapolate_only.png"), bbox_inches = "tight")
    plt.close()

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    datetimes = list(file_data["date"]) 
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)
 
    for model_name in os.listdir("train_net/" + filename_no_csv + "/predictions/test"):  
            
        if not os.path.isdir("train_net/" + filename_no_csv + "/plots/" + model_name):
            os.makedirs("train_net/" + filename_no_csv + "/plots/" + model_name) 
        
        for filename in os.listdir("train_net/" + filename_no_csv + "/predictions/test/" + model_name): 
            
            test_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/test/" + model_name + "/" + filename, index_col = False, sep = ";") 
            
            ytest = list(test_data["actual"])
            predict_test = list(test_data["predicted"]) 

            ws = filename.replace(".csv", "").split("_")[-4] 
            hidden = filename.replace(".csv", "").split("_")[-2] 
        
            train_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/train/" + model_name + "/" + filename.replace("test", "train"), index_col = False, sep = ";") 
            
            ytrain = list(train_data["actual"])
            predict_train = list(train_data["predicted"])
            
            val_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/validate/" + model_name + "/" + filename.replace("test", "validate"), index_col = False, sep = ";") 
            
            yval = list(val_data["actual"])
            predict_val = list(val_data["predicted"])
                
            extrapolate_data = pd.read_csv("train_net/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename.replace("test", "extrapolate"), index_col = False, sep = ";") 
             
            dates_strings = list(extrapolate_data["dates"])
            predict_extrapolate = list(extrapolate_data["predicted"])
 
            extrapolate_data_longer = pd.read_csv("train_net/" + filename_no_csv + "/predictions/extrapolate/" + model_name + "/" + filename.replace("test", "extrapolate_longer"), index_col = False, sep = ";") 
             
            dates_strings_longer = list(extrapolate_data_longer["dates"])
            predict_extrapolate_longer = list(extrapolate_data_longer["predicted"])
            
            is_a_nan = False

            for val in predict_train:
                if str(val) == 'nan':
                    is_a_nan = True
                    break
                 
            for val in predict_val:
                if str(val) == 'nan':
                    is_a_nan = True
                    break
                 
            for val in predict_test:
                if str(val) == 'nan':
                    is_a_nan = True
                    break

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
                plot_result(predict_extrapolate, dates_strings, ytrain, yval, ytest, predict_train, predict_val, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "train_net/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + ".png")
                plot_result(predict_extrapolate_longer, dates_strings_longer, ytrain, yval, ytest, predict_train, predict_val, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", datetimes, "train_net/" + filename_no_csv + "/plots/" + model_name + "/" + filename_no_csv + "_" + model_name + "_ws_" + str(ws) + "_hidden_" + str(hidden) + "_longer.png")

                train_RMSE, val_RMSE, test_RMSE = print_error(ytrain, yval, ytest, predict_train, predict_val, predict_test, "Visina površine mora predviđena " + model_name + " modelom (veličina prozora " + str(ws) + ", " + str(hidden) + " skrivenih slojeva)", range_val)