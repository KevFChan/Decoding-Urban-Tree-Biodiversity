#This script runs the upsampling code to process the SDC 500m data for a single city

#Import packages
import os
import rasterio
from rasterio.enums import Resampling
from concurrent.futures import ProcessPoolExecutor
import argparse
from affine import Affine

#We will now define the upsampling function
def upsample_with_rasterio(args):
    #Define the input, output, and upscale factor arguments
    input_path, output_path, upscale_factor = args

    #Open the path
    with rasterio.open(input_path) as dataset:
        data = dataset.read(
            out_shape = (
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            #Perform nearest neighbors resampling
            resampling = Resampling.nearest
        )

        #Now we transform the data set and write it to the output_path
        #Note: This might not work
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        # #An alternative method
        # scale_x = dataset.width / dataset.shape[-1]
        # scale_y = dataset.height / dataset.shape[-2]
        # transform = dataset.transform * Affine.scale(scale_x, scale_y)

        #Now we write the dataset
        with rasterio.open(output_path, 'w',
                           driver = 'GTiff',
                           height = data.shape[1],
                           width = data.shape[2],
                           count = dataset.count,
                           dtype = str(data.dtype),
                           crs = dataset.crs,
                           transform = transform) as dest:
            dest.write(data)

    print(f"Processed {input_path} to {output_path}")


    #Define the main function
def main():
    #Parse the directories from the wrapper function
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required = True, help = "The input directory for the data for this processing")
    parser.add_argument("--output_directory", required = True, help = "The output directory for the data after we have upsampled")
    args = parser.parse_args()

    upscale_factor = 16
    # input_directory = r'C:\Temp\MODIS\SDC500\temp' #Pre_Fusion_Crop_full'
    # output_directory = r'C:\Temp\MODIS\SDC500\Pre_Fusion_Upsample_Boston'
    input_directory = args.input_directory
    output_directory = args.output_directory

    os.makedirs(output_directory, exist_ok = True)

    #Define the target files on which we will upsample
    target_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.lower().endswith('.tif')]

    #Define the tasks we will submit for parallel processing
    tasks = [(tgt, os.path.join(output_directory, os.path.basename(tgt).replace('prefusion_', 'upsampled_')), upscale_factor) for tgt in target_files]

    #Submit the job
    with ProcessPoolExecutor(max_workers = 20) as executor:
        results = list(executor.map(upsample_with_rasterio, tasks))

    print(f"SDC500 upsampling completed.")

if __name__ == "__main__":
    main()