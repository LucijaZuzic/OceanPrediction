import pandas as pd
import numpy as np
import os
import math
from sklearn.metrics import mean_squared_error
from scipy import fft
import matplotlib.pyplot as plt

for filename in os.listdir("processed"):

    file_data = pd.read_csv("processed/" + filename, index_col = False, sep = ";")

    filename_no_csv = filename.replace(".csv", "")
     
    x = list(file_data["sla"][:100])  
    x = [np.sin(x) for x in np.arange(0, 10, np.pi / 36)] 

    plt.plot(x)  

    plt.show()
    plt.close()

    X = fft.fft(x) 

    plt.plot(fft.fftfreq(len(x)), X.real)  
    plt.show()
    plt.close()

    plt.plot(fft.fftfreq(len(x)), X.imag)  
    plt.show()
    plt.close()

    ampl = []
   
    for k in range(len(X)):

        cos_vals = [x[n] * np.cos(2 * np.pi * k * n / len(x)) for n in range(len(x))]
        sin_vals = [x[n] * np.sin(2 * np.pi * k * n / len(x)) for n in range(len(x))]
   
        ampl.append(np.sqrt(np.sum(cos_vals) ** 2 + np.sum(sin_vals) ** 2))
  
    plt.plot(fft.fftfreq(len(x)), ampl)  
    plt.show()
    plt.close()

    print(max(X.real), np.argmax(X.real), fft.fftfreq(len(x))[np.argmax(X.real)], fft.fftfreq(len(x))[np.argmax(X.real)] * np.pi, fft.fftfreq(len(x))[np.argmax(X.real)] * 2 * np.pi)
    print(max(X.imag), np.argmax(X.imag), fft.fftfreq(len(x))[np.argmax(X.imag)], fft.fftfreq(len(x))[np.argmax(X.imag)] * np.pi, fft.fftfreq(len(x))[np.argmax(X.imag)] * 2 * np.pi)
    print(max(ampl), np.argmax(ampl), fft.fftfreq(len(x))[np.argmax(ampl)], fft.fftfreq(len(x))[np.argmax(ampl)] * np.pi, fft.fftfreq(len(x))[np.argmax(ampl)] * 2 * np.pi)
   
    break