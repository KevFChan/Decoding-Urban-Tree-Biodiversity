#The script runs the image fusion code on the Landsat_ARD and MODIS data to perform cuFSDAF

#Import packages
import os
import rasterio
import numpy as np
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import argparse


def extract_date_from_filename(filename):
    """Extracts date from filename in the format 'stacked_YYYYMMDD_sensor.tif'."""
    date_str = os.path.basename(filename).split('_')[1]
    return datetime.strptime(date_str, "%Y%m%d")

def create_date_file_dict(file_list):
    """Create a dictionary with dates as keys and filenames as values."""
    date_file_dict = {}
    for file in file_list:
        date = extract_date_from_filename(file)
        date_file_dict[date] = file
    
    return date_file_dict

def find_closest_date(date, dates_list):
    """Find the closest date in a list of dates."""
    return min(dates_list, key = lambda d: abs(d - date))

def generate_param_file(output_file, path_class, f1, c1, c2, template_file, scale):
    """
    Generates a new parameter file with replaced paths for f1, c1, and c2.
    """
    with open(template_file, 'r') as template:
        content = template.read()

        # Replace the paths in the content
        replacements = {
            'IN_F1_NAME = \AHB\Landsat\L_2014-4-15.tif': f"IN_F1_NAME = {f1}",
            'IN_C1_NAME = \AHB\MODIS\M_2014-4-15.tif': f"IN_C1_NAME = {c1}",
            'IN_C2_NAME = \AHB\MODIS\M_2014-9-6.tif': f"IN_C2_NAME = {c2}",
            'IN_F1_CLASS_NAME = \AHB\class_L_2015-6-21.tif': f"IN_F1_CLASS_NAME = {os.path.join(path_class,os.path.basename(f1).replace('clear_','classified_'))}",
            'SCALE_FACTOR = 16' : f"SCALE_FACTOR = {scale}",
            'IDW_SEARCH_RADIUS = 16' : f"IDW_SEARCH_RADIUS = {scale}"
            # ... You can add more replacements as necessary
        }

        for old, new in replacements.items():
            content = content.replace(old, new)
    
    with open(output_file, 'w') as output:
        output.write(content)


def run_cuFSDAF(executable_path, parameter_file):
    """
    Run cuFSDAF with the specified parameter file.
    """
    cmd = [executable_path, parameter_file]

    result = subprocess.run(cmd, capture_output = True, text = True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)


#Now define the function to process our data
def process_c2_file(c2, f1_dates, c1_dates, f1_date_dict, c1_date_dict, parameter_directory, path_class, template_path, executable_path, scale):
    target_date = extract_date_from_filename(c2)

    #Find the closest dates using the dictionaries
    closest_f1_date = find_closest_date(target_date, f1_dates)
    closest_c1_date = find_closest_date(closest_f1_date, c1_dates)

    #Use the closest dates to get the corresponding file names
    f1 = f1_date_dict[closest_f1_date]
    c1 = c1_date_dict[closest_c1_date]

    formatted_date = target_date.strftime('%Y-%m-%d')
    output_param_file = os.path.join(parameter_directory, f"Parameters_{formatted_date}.txt")
    generate_param_file(output_param_file, path_class, f1, c1, c2, template_path, scale)
    run_cuFSDAF(executable_path, output_param_file)


#Now we define the main function
def main():
    #Parse the directories from the wrapper function
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable_path", required = True, help = "Executable path for cuFSDAF code")
    parser.add_argument("--template_path", required = True, help = "The template path for the parameters of cuFSDAF")
    parser.add_argument("--parameter_directory", required = True, help = "The parameter directory of HLS_MODIS")
    parser.add_argument("--path_class", required = True, help = "The data files that were created in step 5 after we ran the Pre_Fusion data through a Kmeans classifier")
    parser.add_argument("--path_landsat", required = True, help = "The data that was inputted into step 5")
    parser.add_argument("--path_modis", required = True, help = "The MODIS data we upsampled in step 8")
    parser.add_argument("--output_directory", required = True, help = "The output we will write our image fusion files to")
    args = parser.parse_args()

    # executable_path = "/mnt/d/SetoLab/code/Archive/cuFSDAF-master/Code/cuFSDAF"
    # template_path = "/mnt/d/SetoLab/code/Archive/cuFSDAF-master/Code/Parameters_template.txt"
    # parameter_directory = "/mnt/d/SetoLab/code/Archive/cuFSDAF-master/Code/parameter_HLS_MODIS"
    # path_class = "/mnt/d/SetoLab/One30m/Pre_Fusion_Class_full"
    executable_path = args.executable_path
    template_path = args.template_path
    parameter_directory = args.parameter_directory
    path_class = args.path_class

    # path_landsat = f"/mnt/d/SetoLab/One30m/Pre_Fusion_500m_full"
    # path_modis = r"/mnt/c/Temp/MODIS/SDC500/Pre_Fusion_Upsample_full"
    path_landsat = args.path_landsat
    path_modis = args.path_modis
    output_path = args.output_directory
    scale = 16


    #Get the list of all the files to process
    f1_files = [os.path.join(path_landsat, f) for f in os.listdir(path_landsat) if f.lower().endswith('.tif')]
    c1_files = [os.path.join(path_modis, f) for f in os.listdir(path_modis) if f.lower().endswith('.tif')]

    c2_files = c1_files

    #Create the dictionaries for the f1_files and c1_files
    f1_date_dict = create_date_file_dict(f1_files)
    c1_date_dict = create_date_file_dict(c1_files)
    f1_dates = list(f1_date_dict.keys())
    c1_dates = list(c1_date_dict.keys())

    #Parallelize the execution loop
    with ProcessPoolExecutor(max_workers = 3) as executor:
        futures = [
            executor.submit(process_c2_file, 
                            c2, f1_dates, c1_dates, f1_date_dict, c1_date_dict, 
                            parameter_directory, path_class, template_path, executable_path, scale
                            ) for c2 in c2_files
        ]

        for future in futures:
            future.result()
        

if __name__ == "__main__":
    main()