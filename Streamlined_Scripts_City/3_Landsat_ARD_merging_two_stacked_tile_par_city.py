#Import the libraries
import os
import rasterio
from rasterio.merge import merge
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse

#Define the merged tiles
def merge_tiles(args):
    date, sensor, files, path_merged = args

    #Open all files for a given date and sensor
    sources = [rasterio.open(f) for f in files]

    #Merge tiles
    mosaic, out_trans = merge(sources)
    merged_meta = sources[0].meta.copy()
    merged_meta.update({"driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans})
    
    #Write the merged file
    merged_filename = os.path.join(path_merged, f'merged_{date}_{sensor}')
    with rasterio.open(merged_filename, 'w', **merged_meta) as merged_dst:
        merged_dst.write(mosaic)

    #Close all source files
    for src in sources:
        src.close()

    print(f"Merged file for {date}, Sensor {sensor} created")


#Main function
if __name__ == '__main__':
    #Unwrap the inputs here
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required = True, help = f"Input directory of the Landsat ARD files")
    parser.add_argument("--output_directory", required = True, help = f"Output directory of the merged Landsat ARD files")
    parse_args = parser.parse_args()


    #Paths
    # path_stacked = f'F:/Landsat_ARD/Stacked/{city}/SR/'
    # path_merged = f'F:/Landsat_ARD/Merged/{city}/SR/'
    path_stacked = parse_args.input_directory
    path_merged = parse_args.output_directory

    os.makedirs(path_merged, exist_ok = True)

    #Create a dictionary to group filenames by date and sensor
    files_by_date_sensor = defaultdict(lambda: defaultdict(list))

    #List all the stacked files and group them by date and sensor
    for filename in os.listdir(path_stacked):
        if filename.endswith('.tif'):
            parts = filename.split('_')
            date = parts[1]
            #May need to fix this
            sensor = parts[2]
            files_by_date_sensor[date][sensor].append(os.path.join(path_stacked, filename))
    
    #Run parallel
    with ProcessPoolExecutor(max_workers = 5) as executor:
        #Prepare arguments for each merging task
        tasks = [(date, sensor, files, path_merged) for date, sensors in files_by_date_sensor.items() for sensor, files in sensors.items()]

        #Execute the tasks in parallel
        results = list(executor.map(merge_tiles, tasks))
    
    print("All merging completed.")

