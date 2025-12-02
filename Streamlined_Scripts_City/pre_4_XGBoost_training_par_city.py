#The goal of this .py file is to train 6 XGBoost models on Landsat ARD data to predict the HLS data

#Import the packages
import os
import numpy as np
import xgboost as xgb
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib


#Define the main function
def main():
    landsat_directory = "Temp"
    hls_directory = "temp"
    output_directory = "temp"

    #Ensure that the output directory exits
    os.makedirs(output_directory, exist_ok = True)

    #I need to first read the data, then structure it, and pass it through the XGBoost model

    

if __name__ == "__main__":
    main()