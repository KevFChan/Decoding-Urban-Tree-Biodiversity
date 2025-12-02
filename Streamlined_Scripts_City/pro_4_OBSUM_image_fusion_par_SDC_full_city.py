#This script will run the OBSUM code to perform spatiotemporal fusion

#Import packages
import os
import rasterio
import numpy as np
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from skimage.transform import downscale_local_mean, resize
from OBSUM_Functions import OBSUM, read_raster, write_raster

#Initialize the parameters for OBSUM
# Scale factor between coarse and fine image, for HLS and SDC case, our SDC was resampled to 480 meters and HLS is 30 meters
scale_factor = 16
# Number of land-cover classes in the base fine image, we used 8 classes in our K-means clustering
class_num = 8
# Size of the local unmixing window,
# it's recommended to use a large value, e.g., 15
win_size = 15
# Percentage of the selected fine pixels for object-level residual compensation,
# it's recommended to use a small value, e.g., 5
object_RC_percent = 5
# Size of the local window for similar pixels selection
similar_win_size = 31
# Number of similar pixels need to be selected
similar_num = 30
# Min and max value of the data, while the authors recommend 1.0 for the max value, we must multiply by 10,000 to account for our differences in scaling since the values range up to 10,000
min_val = 0.0
max_val = 1.0 * 10000


#Define a function to run the OBSUM
def run_OBSUM(C_tb_path, F_tb_path, F_tb_object_path, F_tb_class_path, C_tp_path, F_tp_OBSUM_path):
    #Load in the files
    F_tb, F_tb_profile = read_raster(F_tb_path)
    C_tb = read_raster(C_tb_path)[0]
    #C_tb_coarse = downscale_local_mean(C_tb, factors = (scale_factor, scale_factor, 1))
    C_tb_coarse = C_tb

    C_tp = read_raster(C_tp_path)[0]
    #C_tp_coarse = downscale_local_mean(C_tp, factors = (scale_factor, scale_factor, 1))
    C_tp_coarse = C_tp

    F_tb_objects = read_raster(F_tb_object_path)[0][:, :, 0]
    F_tb_class = read_raster(F_tb_class_path)[0][:, :, 0]

    # F_tb shape is (1920, 1776, 7)
    # C_tb shape is (120, 111, 6)
    # C_tb_coarse shape is (8, 7, 6)
    # C_tp_coarse shape is (8, 7, 6)
    # F_tb_objects shape is (1920, 1776)
    # F_tb_class shape is (1920, 1776)

    #Perform some shape checks
    assert F_tb.shape[:2] == F_tb_class.shape, "F_tb and F_tb_class shapes do not match!"
    assert F_tb.shape[:2] == F_tb_objects.shape, "F_tb and F_tb_object shapes do not match!"

    #Now we run OBSUM
    time0 = datetime.now()
    obsum = OBSUM(F_tb, C_tb_coarse, C_tp_coarse, F_tb_class, F_tb_objects,
                  class_num = class_num, scale_factor = scale_factor, win_size = win_size,
                  OL_RC_percent = object_RC_percent,
                  similar_win_size = similar_win_size, similar_num = similar_num,
                  min_val = min_val, max_val = max_val)
    
    F_tp_OBSUM = obsum.object_based_spatial_unmixing()
    time1 = datetime.now()
    time_span = time1 - time0
    print(f"Used {time_span.total_seconds():.2f} seconds!")

    #Save the predicted image
    write_raster(F_tp_OBSUM, F_tb_profile, F_tp_OBSUM_path)

def process_OBSUM_input(args):
        try:
            print(f"Processing:\n"
              f"  Prev MODIS : {args[0]}\n"
              f"  HLS        : {args[1]}\n"
              f"  SAM        : {args[2]}\n"
              f"  Class      : {args[3]}\n"
              f"  Curr MODIS : {args[4]}\n"
              f"  Output     : {args[5]}")
            run_OBSUM(*args)
            print(f"Finished processing {args[-1]}")
        except Exception as e:
            print(f"Error processing {args[-1]}: {e}")

#In order to make this code work, I need to segment the image objects in the base fine image (Landsat)
#I need to create a land-cover classification map of the base fine image
#I need to define a tb and a tp time frame

#Define the main function
def main():
    #Parse in the arguments from the wrapper function
    parser = argparse.ArgumentParser()

    #We need five paths, the coarse image directory, the fine image directory, the fine image segmented, the fine image after being classified, and the output path for the fine image
    parser.add_argument("--path_landsat", required = True, help = "The landsat HLS fine resolution data")
    parser.add_argument("--path_MODIS", required = True, help = "The MODIS SDC 500 coarse resolution data")
    parser.add_argument("--path_SAM", required = True, help = "The fine resolution segmented images")
    parser.add_argument("--path_class", required = True, help = "The fine reslution images from the K-means classification step")
    parser.add_argument("--output_directory", required = True, help = "The output directory for the OBSUM-processed images")
    args = parser.parse_args()

    #Now we unpack them
    landsat_fine_dir = args.path_landsat
    modis_coarse_dir = args.path_MODIS
    sam_fine_dir = args.path_SAM
    class_fine_dir = args.path_class
    output_dir = args.output_directory

    #Make sure that the output directory exists
    os.makedirs(output_dir, exist_ok = True)

    #Now we want to match up the files to each other
    landsat_fine_paths = os.listdir(landsat_fine_dir)
    modis_coarse_paths = os.listdir(modis_coarse_dir)
    sam_fine_paths = os.listdir(sam_fine_dir)
    class_fine_paths = os.listdir(class_fine_dir)
    


    ###################### We now want to set up our inputs for OBSUM
    modis_dates = sorted([(f, datetime.strptime(f.split('_')[1], "%Y%m%d")) for f in modis_coarse_paths], key = lambda x: x[1])
    landsat_dates = sorted([(f, datetime.strptime(f.split('_')[1], "%Y%m%d")) for f in landsat_fine_paths], key = lambda x: x[1])

    #Index the SAM and class files by HLS date
    sam_dict = {datetime.strptime(f.split('_')[1], "%Y%m%d"): f for f in sam_fine_paths}
    class_dict = {datetime.strptime(f.split('_')[1], "%Y%m%d"): f for f in class_fine_paths}

    #Convert to dictionaries for faster lookup
    modis_dict = {date: f for f, date in modis_dates}
    landsat_dict = dict(landsat_dates)

    #Build a set of the existing HLS dates
    landsat_dates_set = set(date for _, date in landsat_dates)

    OBSUM_inputs = []

    for curr_modis_file, curr_date in modis_dates:
        #Ensure that we are skipping the dates we have fine resolution data for
        if curr_date in landsat_dates_set:
            continue
        
        #Find the latest HLS file before or on the current MODIS date
        matching_landsat = None
        for landsat_file, landsat_date in reversed(landsat_dates):
            if landsat_date <= curr_date:
                matching_landsat = (landsat_file, landsat_date)
                break

        #Find the MODIS file for that previous date and get the SAM and classified_fine_resolution files
        if matching_landsat:
            prev_date = matching_landsat[1]
            prev_modis_file = os.path.join(modis_coarse_dir, modis_dict.get(prev_date))
            sam_file = os.path.join(sam_fine_dir, sam_dict.get(prev_date))
            class_file = os.path.join(class_fine_dir, class_dict.get(prev_date))
            curr_modis_file = os.path.join(modis_coarse_dir, curr_modis_file)

            #Create the output path
            curr_fine_filename = f"OBSUM_processed_{curr_date.strftime("%Y%m%d")}_landsatARD.tif"
            curr_fine_path = os.path.join(output_dir, curr_fine_filename)

            if prev_modis_file:
                OBSUM_inputs.append((prev_modis_file,
                                     os.path.join(landsat_fine_dir, matching_landsat[0]),
                                     sam_file,
                                     class_file,
                                     curr_modis_file,
                                     curr_fine_path))
    
    # for prev_modis, hls_file, sam_file, class_file, curr_modis, curr_fine_path in OBSUM_inputs:
    #     print(f"Prev Modis: {prev_modis}, HLS: {hls_file}, SAM: {sam_file}, Class: {class_file}, Current MODIS: {curr_modis}, Output path: {curr_fine_path}")
    
    ###################### Now we run OBSUM in parallel

    with ProcessPoolExecutor(max_workers = 10) as executor:
        futures = [executor.submit(process_OBSUM_input, obs_input) for obs_input in OBSUM_inputs]

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
