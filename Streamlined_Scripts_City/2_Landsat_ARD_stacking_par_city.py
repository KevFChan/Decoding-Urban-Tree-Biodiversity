import os
import rasterio
from collections import defaultdict
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import argparse

def nested_dict():
    return defaultdict(list)

def nested_dict_2():
    return defaultdict(nested_dict)

def stack_bands(args, path_output):
    date, tiles_dict = args

    for tile_id, sensor_files_dict in tiles_dict.items():
        for sensor, files in sensor_files_dict.items():
            #Sort files by band number
            files = sorted(files, key = lambda x: x.split('.')[-2])

            #Open the first file to get the metadata
            with rasterio.open(files[0]) as src0:
                meta = src0.meta

            #Update the meta to reflect the number of layers
            meta.update(count = len(files), dtype = "uint16")

            #Write the stacked bands to a new file
            stacked_filename = os.path.join(path_output, f'stacked_{date}_{tile_id}_{sensor}.TIF')
            with rasterio.open(stacked_filename, 'w', **meta) as dst:
                for id, layer in enumerate(files, start = 1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(id, src1.read(1).astype('uint16'))
            
            print(f"Stacking for {date}, Tile {tile_id}, Sensor {sensor} completed.")


if __name__ == '__main__':
    #Define the paths by parsing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required = True, help = f"Input directory of the extracted .TAR files")
    parser.add_argument("--output_directory", required = True, help = f"Output directory of the stacked data")
    parse_args = parser.parse_args()

    input_directory = parse_args.input_directory
    output_directory = parse_args.output_directory
    os.makedirs(output_directory, exist_ok=True)

    files_by_date_tile_sensor = defaultdict(nested_dict_2)

    for filename in os.listdir(input_directory):
        if filename.endswith('.TIF'):
            parts = filename.split('_')
            date = parts[3]
            sensor = parts[0]
            tile_id = parts[2]
            files_by_date_tile_sensor[date][tile_id][sensor].append(os.path.join(input_directory, filename))
    
    #Now use the ProcessPoolExecutor to speed up stacking
    with ProcessPoolExecutor(max_workers = 5) as executor:
        stack_bands_with_path = partial(stack_bands, path_output = output_directory)
        results = list(executor.map(stack_bands_with_path, files_by_date_tile_sensor.items()))

    for r in results:
        print(r)
    
    print("All band stacking complete")