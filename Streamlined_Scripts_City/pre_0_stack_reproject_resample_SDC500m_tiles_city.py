#The script stacks, reprojects, and resamples the SDC500m data for our purposes

#Import the packages
import os
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
from rasterio.crs import CRS
from osgeo import gdal
from concurrent.futures import ProcessPoolExecutor
import shutil
import argparse
from pyproj import Proj

gdal.UseExceptions()

#We will first define the functions we will use to stack the data
def stack_bands(date_files_pair, output_dir):
    #Takes in a date_files pair which contains both the date and the file list directory for us to loop through, as well as an output directory for us to send the results to
    date, file_list = date_files_pair

    #Define the desired order of the bands
    desired_order = [3, 4, 1, 2, 5, 6]

    #Reorder the file_list based on the desired order of the bands
    #Extract band numbers from the filenames and match them to the desired order for sorting
    file_list_sorted = sorted(file_list, key = lambda x: desired_order.index(int(x.split('_')[-1].split('.')[0][1:])))

    #Adjust the sorting logic for the specific filename format where instances contain "_b"
    file_list_sorted = sorted(file_list, key = lambda x: desired_order.index(int(x.split('_b')[-1].split('.')[0])))

    #Read the metadata from the first file since the metadata of the other files are in the same format
    with rasterio.open(file_list_sorted[0]) as src0:
        meta = src0.meta
    
    #Update meta to reflect the number of layers and datatype we want
    meta.update(count = len(file_list_sorted), dtype = 'int16')

    #Write the stacked bands to a new file in the desired order
    output_path = os.path.join(output_dir, f'stacked_{date}_MODIS.tif')

    #Write the raster file to that output path after converting the first band of the raster file into int16
    with rasterio.open(output_path, 'w', **meta) as dst:
        for id, layer in enumerate(file_list_sorted, start = 1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1).astype('int16'))
    
    return f"Stacking for {date} completed."


#Define the stack_helper function for when we call it in the parallel processing setup
def stack_helper(args):
    return stack_bands(*args)


#Define the reprojection function
def reproject_MODIS(args):
    """
    Reprojects and resamples a raster dataset.
    
    Parameters:
    - input_path: path to the input raster file
    - output_path: path where the reprojected and resampled raster will be saved
    - new_crs: the new coordinate reference system (can be an EPSG code or a proj4 string)
    - new_resolution: the new resolution for the raster
    """

    input_path, output_path, new_crs = args

    #Define the CRS for MODIS Sinusoidal
    #src_crs = 'PROJCS["MODIS Sinusoidal",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Sinusoidal"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",0.0],PARAMETER["semi_major",6371007.181],PARAMETER["semi_minor",6371007.181],UNIT["m",1.0],AUTHORITY["SR-ORG","6974"]]'
    #This should be equivalent to the above
    src_crs = CRS.from_proj4("+proj=sinu +R=6371007.181 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
    
    with rasterio.open(input_path) as src:
        #Use the source GeoTIFF file
        #src_crs = src_crs

        #Calculate the transformation we will be making with the reprojection
        transform, width, height = calculate_default_transform(
            src_crs, new_crs, src.width, src.height, *src.bounds
        )

        #Update the arguments
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': new_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        #Now we reproject the files to the output path
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source = rasterio.band(src, i),
                    destination = rasterio.band(dst, i),
                    src_transform = src.transform,
                    src_crs = src_crs,
                    src_nodata = None,
                    dst_transform = transform,
                    dst_crs = new_crs,
                    dst_nodata = None,
                    resampling = Resampling.cubic_spline
                )


#Create a function to call the reprojecton function above (makes the main function cleaner)
def reproject_data(input_directory, output_directory, new_crs):
    #Create a list of files to process by including both the current path as a "extracted" raster and its new path once it is reprojected
    files_to_process = [(os.path.join(input_directory, file),
                         os.path.join(output_directory, f"{file.replace('stacked', 'reprojected')}"),
                         new_crs)
                         for file in os.listdir(input_directory) if file.lower().endswith('.tif') and 'stacked' in file]


    #Get the number of CPUs for parallel processing
    num_cpus = cpu_count()

    #Now parallelize the function
    with Pool(processes = num_cpus) as pool:
        pool.map(reproject_MODIS, files_to_process)


#Now we create the resampling function
def resample_MODIS(args):
    input_path, output_path, new_resolution = args
    x_res = new_resolution[0]
    y_res = new_resolution[1]

    gdal.Warp(output_path, input_path, xRes = x_res, yRes = y_res, resampleAlg = gdal.GRA_CubicSpline)


#Create a function to call the resample function above (makes the main function cleaner)
def resample_data(input_directory, output_directory, new_resolution):
    #Create a list of files to process by including both the current path as a "reprojected" raster and its new path once it is resampled
    files_to_process = [(os.path.join(input_directory, file),
                         os.path.join(output_directory, f"{file.replace('reprojected', 'resampled')}"),
                         new_resolution)
                         for file in os.listdir(input_directory) if file.lower().endswith('.tif')]
    
    #Get the number of CPUs for parallel processing
    num_cpus = cpu_count()

    #Now parallelize the function
    with Pool(processes = num_cpus) as pool:
        pool.map(resample_MODIS, files_to_process)
    

#Create a function to delete folders and directories after we are done with using them
def delete_folder(folder_path):
    try:
        #Delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting folder: {str(e)}")


#Now create the main function
def main():
    #Parse in the arguments from the wrapper function
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required = True, help = "Directory for the TIFF files before stacking")
    parser.add_argument("--stacked_directory", required = True, help = "Output directory for the stacked tiff files")
    parser.add_argument("--reprojected_directory", required = True, help = "Output directory for the reprojected raster data")
    parser.add_argument("--resampled_directory", required = True, help = "Output directory for the resampled raster files")
    parser.add_argument("--longitude", required = True, help = "Longitude of the city")
    parser.add_argument("--latitude", required = True, help = "Latitude of the city")
    args = parser.parse_args()

    #Now we assign the variables
    base_path = args.input_directory
    stacked_path = args.stacked_directory
    reprojected_path = args.reprojected_directory
    resampled_path = args.resampled_directory
    lon = float(args.longitude)
    lat = float(args.latitude)

    #Define the tiles
    #Now we will convert the latitude and longitude values into its corresponding MODIS tile by using pyproj
    #Define the upper-left corner X, upper-left corner Y, and tile size
    WORLD_ULC_X = -20015109.354
    WORLD_ULC_Y = 10007554.677
    TILE_SIZE = 1111950

    #Define the MODIS sinusoidal projection
    modis_proj = Proj("+proj=sinu +R=6371007.181 +lon_0=0")
    x, y = modis_proj(lon, lat)

    #Compute h and v
    h = int((x - WORLD_ULC_X) / TILE_SIZE)
    v = int((WORLD_ULC_Y - y) / TILE_SIZE)

    #Define the tile using the sinusoidal definition
    tile = f"h{h:02d}v{v:02d}"

    #tiles = ['h10v06']#['h11v04','h11v05','h12v04','h12v05']

    #Define the steps for stacking, reprojecting, and resampling for each tile
    #Step 1: Stack the TIFFs
    # base_path_directory = f'F:/SDC/{tile}'
    # stacked_directory = f'F:/SDC_Stacked/{tile}'
    base_path_directory = base_path
    stacked_directory = stacked_path

    #We do not need to stack our bands right now so we just read in the stacked datasets and run the following code
    stacked_directory = base_path_directory
    
    # #Check if the directory exists, if not then create it
    # if not os.path.exists(stacked_directory):
    #     os.makedirs(stacked_directory)
    
    # files_by_date = defaultdict(list)

    # #Now we loop through the base path directory
    # for filename in os.listdir(base_path_directory):
    #     if filename.endswith('.tif'):
    #         parts = filename.split('_')
    #         year_doy = parts[1]
    #         year, doy = int(year_doy[:4]), int(year_doy[4:])
    #         #Calculate the date
    #         date = datetime(year, 1, 1) + timedelta(doy - 1)
    #         date = date.strftime('%Y%m%d')
    #         files_by_date[date].append(os.path.join(base_path_directory, filename))
    
    # # #Now we loop through the base path directory
    # # for year_dir in os.listdir(base_path_directory):
    # #     year_path = os.path.join(base_path_directory, year_dir)

    # #     if os.path.isdir(year_path):
    # #         for filename in os.listdir(year_path):
    # #             if filename.endswith('.TIF'):
    # #                 parts = filename.split('_')
    # #                 year_doy = parts[1] 
    # #                 year, doy = int(year_doy[:4]), int(year_doy[4:])
    # #                 #Calculate the date
    # #                 date = datetime(year, 1, 1) + timedelta(doy - 1)
    # #                 date = date.strftime('%Y%m%d')
    # #                 files_by_date[date].append(os.path.join(year_path, filename))

    # #Define the parallel processing loop
    # with ProcessPoolExecutor(max_workers = 5) as executor:
    #     tasks = [(date_files, stacked_directory) for date_files in files_by_date.items()]
    #     results = executor.map(stack_helper, tasks)
    
    # for r in results:
    #     print(r)
    
    # print("All band stacking completed.")

    #Now we perform Step 2: Reprojection
    # reprojected_directory = f'E:/SDC_Reprojected/{tile}'
    reprojected_directory = reprojected_path
    new_crs = CRS.from_epsg(32618) # Example: UTM Zone 18N

    #Check if the directory exists, if not then create it
    if not os.path.exists(reprojected_directory):
        os.makedirs(reprojected_directory)

    #Reproject the data
    reproject_data(stacked_directory, reprojected_directory, new_crs)

    print("Reprojection completed.")
    #Delete the stacked directory folder and its contents since we no longer need it
    #delete_folder(stacked_directory)

    #Now we perform Step 3: Resampling
    # resampled_directory = f'E:/SDC_Resampled/{tile}'
    resampled_directory = resampled_path
    new_resolution = (480, 480)

    #Check if the directory exists, if not then create it
    if not os.path.exists(resampled_directory):
        os.makedirs(resampled_directory)
    
    #Resample
    resample_data(reprojected_directory, resampled_directory, new_resolution)

    print("Resampling completed.")
    

#Main function
if __name__ == '__main__':
    main()

