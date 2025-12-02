#The script runs the crop, fusion, and extend processes on the LandsatARD and MODIS data. 
#It uses the data generated from our “pre_0_1_reprojection_band_extraction_LandsatARD.py” code

#Import packages
import os
import re
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import argparse

#We first want to get the overlapping windows between two raster datasets
def get_overlapping_window(src1, src2):
    """
    Determines the overlapping window of two raster datasets.
    Returns the window for both rasters as a tuple.
    """
    #Get the bounds of both datasets
    left1, bottom1, right1, top1 = src1.bounds
    left2, bottom2, right2, top2 = src2.bounds

    #Determine the overlapping bounds
    left, bottom, right, top = max(left1, left2), max(bottom1, bottom2), min(right1, right2), min(top1, top2)

    #If there is no overlap, then return None
    if left >= right or bottom >= top:
        return None, None
    
    #Else we will get the window of the overlap for each dataset
    window1 = src1.window(left, bottom, right, top)
    window2 = src2.window(left, bottom, right, top)

    return window1, window2

#Now we will define a function to extract the overlapping regions between two datasets
def extract_overlap_from_datasets(reference_path, target_path, target_out_path):
    """Extracts overlapping regions from two raster datasets."""
    with rasterio.open(reference_path) as ref_src, rasterio.open(target_path) as tgt_src:

        #Get the overlapping window
        window1, window2 = get_overlapping_window(ref_src, tgt_src)

        #If there is no overlap, skip the processing
        if window1 is None or window2 is None:
            print(f"No overlap found between {reference_path} and {target_path}")
            return

        #Else read the overlapping region data directly without resampling
        tgt_data = tgt_src.read(window = window2)

        #Update the metadata
        meta2 = tgt_src.meta.copy()
        meta2.update({
            'width': window2.width,
            'height': window2.height,
            'transform': rasterio.windows.transform(window2, tgt_src.transform)
        })

        #Save the overlapping regions
        with rasterio.open(target_out_path, 'w', **meta2) as tgt_out:
            tgt_out.write(tgt_data)


def process_target_file(reference_file, tgt_file, output_directory_target, current_string, replacement_string):
    #current_string is the string we want to be replaced and replacement_string is the string we want to replace the current string with
    tgt_out_name = os.path.basename(tgt_file).replace(current_string, replacement_string)
    tgt_out_path = os.path.join(output_directory_target, tgt_out_name)

    #Extract the overlapping areas
    extract_overlap_from_datasets(reference_file, tgt_file, tgt_out_path)
    print(f"Processed prefusion for {tgt_file}")



#Now for the main function
#We will perform the LandsatARD tasks first 
def main():
    #Parse in the arguments from the wrapper function
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_directory", required = True, help = "Reference Directory")
    parser.add_argument("--target_ARD_directory", required = True, help = "The directory where the reprojected and extracted Landsat ARD data is")
    parser.add_argument("--output_ARD_directory", required = True, help = "The directory where we will write our crop, fusion, extend processed ARD data to")
    parser.add_argument("--target_MODIS_directory", required = True, help = "The directory where the resampled MODIS data is")
    parser.add_argument("--output_MODIS_directory", required = True, help = "The directory where we will write our crop, fusion, and extend processed MODIS data to")
    parser.add_argument("--city", required = True, help = "The name of the city")
    args = parser.parse_args()

    #Now we initialize the variables/paths
    # reference_directory = r"D:\SetoLab\code\Processing"
    reference_directory = args.reference_directory

    #Paths for Landsat ARD
    # target_directory_ARD = r"C:\Temp\Landsat_ARD\Reprojected_Extracted\city"
    # output_directory_target_ARD = r"D:\SetoLab\Landsat_ARD\Pre_Fusion_Crop_full\city"
    target_directory_ARD = args.target_ARD_directory
    output_directory_target_ARD = args.output_ARD_directory

    #Paths for MODIS
    # city = 'Tampa'
    # target_directory_MODIS = r"C:\Temp\MODIS\Resampled\city"
    # output_directory_target_MODIS = f"C:/Temp/MODIS/SDC500/Pre_Fusion_Crop\city"
    target_directory_MODIS = args.target_MODIS_directory
    output_directory_target_MODIS = args.output_MODIS_directory

    #Get the city
    city = args.city
    
    #Ensure output directories exist
    os.makedirs(output_directory_target_ARD, exist_ok = True)
    os.makedirs(output_directory_target_MODIS, exist_ok = True)

    #Now we create a list of files we want to process for ARD
    reference_files_ARD = [os.path.join(reference_directory, f) for f in os.listdir(reference_directory) if f.endswith(f'Extend_Fusion_{city}_full.tif')]
    target_files_ARD = [os.path.join(target_directory_ARD, f) for f in os.listdir(target_directory_ARD) if f.lower().endswith('.tif')]

    #Now create a list of files we want to process for MODIS
    reference_files_MODIS = [os.path.join(reference_directory, f) for f in os.listdir(reference_directory) if f.endswith(f'Extend_Fusion_{city}_MODIS.tif')]
    target_files_MODIS = [os.path.join(target_directory_MODIS, f) for f in os.listdir(target_directory_MODIS) if f.lower().endswith('.tif')]

    #Check that there is only one reference file for both ARD and MODIS
    if len(reference_files_ARD) != 1 or len(reference_files_MODIS) != 1:
        raise ValueError("There should be only one reference file")
    
    reference_file_ARD = reference_files_ARD[0]
    reference_file_MODIS = reference_files_MODIS[0]


    #Initialize the number of CPUs we will be using
    num_cpus = cpu_count()

    #Parallelize the processing of target files using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers = num_cpus) as executor:
        list(executor.map(process_target_file,
                          [reference_file_ARD] * len(target_files_ARD),
                          target_files_ARD,
                          [output_directory_target_ARD] * len(target_files_ARD),
                          ['reprojected'] * len(target_files_ARD), 
                          ['cropped'] * len(target_files_ARD)))
    
    #Now perform multi-processing for the MODIS data
    with Pool(num_cpus) as pool:
        pool.starmap(process_target_file, [(reference_file_MODIS, tgt, output_directory_target_MODIS, 'resampled', 'prefusion') for tgt in target_files_MODIS])

    #An alternative way to parallelize the MODIS data
    # #Parallelize the processing of the MODIS data
    # with ProcessPoolExecutor(max_workers = num_cpus) as executor:
    #     list(executor.map(process_target_file,
    #                       [reference_file_MODIS] * len(target_files_MODIS),
    #                       target_files_MODIS,
    #                       [output_directory_target_MODIS] * len(target_files_MODIS),
    #                       ['resampled'] * len(target_files_MODIS),
    #                       ['prefusion'] * len(target_files_MODIS)))

if __name__ == '__main__':
    main()