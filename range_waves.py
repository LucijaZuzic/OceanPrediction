import numpy as np
import pandas as pd
import os   
from sklearn.metrics import mean_squared_error
import math

num_props = 1    

dict_for_table = dict()
model_names = set()

for filename_no_csv in os.listdir("train_net"):  

    file_data = pd.read_csv("processed/" + filename_no_csv + ".csv", index_col = False, sep = ";")  
    wave_heights = list(file_data["sla"]) 
    range_val = max(wave_heights) - min(wave_heights)

    print(filename_no_csv.replace("_", " & "), "&", max(wave_heights), "&", min(wave_heights), "&", np.round(range_val, 4), "\\\\ \\hline")