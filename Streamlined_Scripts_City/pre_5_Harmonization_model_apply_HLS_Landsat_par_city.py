#This script applies the XGBoost Regression on the HLS data after the previous step of pro_0_crop_fusion_extend has ran

#Import the packages
import os
import xgboost as xgb
import rasterio
from datetime import datetime
from itertools import repeat
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import argparse


def apply_model_to_band(band_data, model_path):
    """
    Apply the XGBoost model to the given band data.
    """
    #Reshape the band data for prediction
    original_shape = band_data.shape
    flattened_data = band_data.reshape(-1, 1) #Convert a 2D array of n by m into a 2D array of 1 by n*m

    dmatrix = xgb.DMatrix(flattened_data)
    #Initialize the model object
    model = xgb.Booster()
    #Load in the XGBoost model
    model.load_model(model_path)
    #Feed our data into the model for prediction
    predictions = model.predict(dmatrix)

    #Reshape the predictions back into the original band shape
    harmonized_band_data = predictions.reshape(original_shape)

    return harmonized_band_data

#Create a helper function to get the XGBoost model path for a specific band
def find_model_for_band(model_dir, band_num):
    #Searches for the correct model given a band number
    for filename in os.listdir(model_dir):
        if filename.endswith("json") and f"band_{band_num}" in filename:
            return os.path.join(model_dir, filename)
    
    #Else return none
    return None


#Harmonize the dataset
def harmonize_dataset(dataset_path, model_dir, output_directory, coeffs):
    with rasterio.open(dataset_path) as src:
        #Get the meta data for the bands
        meta = src.meta
        harmonized_data = []
        
        for band_num in range(1, meta['count'] + 1):
            band_data = src.read(band_num)
            #Apply transformations
            band_data = band_data*0.0000275 - 0.2

            #Get the model that corresponds to the band
            model_path = find_model_for_band(model_dir, band_num)

            #Apply harmonization to all but the last band
            if band_num < meta['count']:
                coeff_1 = coeffs['coeff_1'][band_num - 1]
                coeff_2 = coeffs['coeff_2'][band_num - 1]
                band_data = band_data * coeff_1 + coeff_2

                harmonized_band_data = apply_model_to_band(band_data, model_path)
                harmonized_band_data = harmonized_band_data * 10000
                harmonized_band_data = harmonized_band_data.astype(np.int16) #Correct data type conversion
                harmonized_data.append(harmonized_band_data)
            else:
                #We can just append the last band without harmonization
                last_band_data = src.read(meta['count'])
                last_band_data = last_band_data.astype(np.int16)
                harmonized_data.append(last_band_data)
        
        #Construct the output filename
        sensor_name = os.path.basename(dataset_path).split('_')[0]
        output_filename = os.path.basename(dataset_path).replace('cropped', 'harmonized')
        output_path = os.path.join(output_directory, output_filename)

        #Update meta data to reflect the number of bands and dtype
        meta.update(count = len(harmonized_data), dtype = "int16")

        #Save the harmonized data
        with rasterio.open(output_path, 'w', **meta) as dst:
            for band_num, band in enumerate(harmonized_data, 1):
                dst.write(band, band_num)

    return f"Processed {dataset_path}"


#Execute the harmonization
if __name__ == "__main__":
    #Get the paths from the argument wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required = True, help = "The input directory of Pre_Fusion_Crop landsat data")
    parser.add_argument("--models", required = True, help = "The directory of the models")
    parser.add_argument("--output_directory", required = True, help = "The output directory after we have harmonized the landsat data")
    args = parser.parse_args()

    # input_dir = r"D:\SetoLab\Landsat_ARD\Pre_Fusion_Crop_full"
    # output_dir = r"D:\SetoLab\Landsat_ARD\Pre_Fusion_Harmonized_full"
    input_dir = args.input_directory
    model_dir = args.models
    output_dir = args.output_directory

    #Confirm that our output directory exists, if not then we make one
    os.makedirs(output_dir, exist_ok = True)

    coeffs = {
        'coeff_1': [0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071],
        'coeff_2': [0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]
    }

    #Get all of the data sets in our folder
    dataset_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

    #Parallel process using the ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers = 12) as executor:
        results = list(executor.map(harmonize_dataset, dataset_paths, repeat(model_dir), repeat(output_dir), repeat(coeffs)))
    
    for result in results:
        print(result) 
