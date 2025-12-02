#The script extracts the overlapping windows between the HLS data and the Landsat_ARD data 

#Import the packages
import os
import re
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from datetime import datetime, timedelta
import argparse


def get_overlapping_window(src1, src2):
    """
    Determines the overlapping window of two raster datasets.
    Returns the window for both rasters as a tuple.
    """
    # Get the bounds of both datasets
    left1, bottom1, right1, top1 = src1.bounds
    left2, bottom2, right2, top2 = src2.bounds
    
    # Determine the overlapping bounds
    left, bottom, right, top = max(left1, left2), max(bottom1, bottom2), min(right1, right2), min(top1, top2)
    
    # If there's no overlap, return None
    if left >= right or bottom >= top:
        return None, None

    # Get the window of the overlap for each dataset
    window1 = src1.window(left, bottom, right, top)
    window2 = src2.window(left, bottom, right, top)
    
    return window1, window2

def resample_to_match(src, target_resolution):
    """Resample a dataset to match a target spatial resolution."""
    new_shape = (
        int(src.height * src.res[0] / target_resolution[0]),
        int(src.width * src.res[1] / target_resolution[1])
    )
    
    data = src.read(
        out_shape=(src.count,) + new_shape,
        resampling=Resampling.bilinear
    )

    transform = src.transform * src.transform.scale(
        src.width / data.shape[-1],
        src.height / data.shape[-2]
    )

    return data, transform

#HLS is the ground truth
def extract_overlap_from_datasets(reference_path, target_path, reference_out_path, target_out_path):
    """Extracts overlapping regions from two raster datasets."""
    with rasterio.open(reference_path) as ref_src, rasterio.open(target_path) as tgt_src:
        
        window1, window2 = get_overlapping_window(ref_src, tgt_src)

        # If there's no overlap, skip further processing
        if window1 is None or window2 is None:
            print(f"No overlap found between {reference_path} and {target_path}.")
            return

        # Read the overlapping region data directly without resampling
        ref_data = ref_src.read(window=window1)
        tgt_data = tgt_src.read(window=window2)

        # Update metadata
        meta1 = ref_src.meta.copy()
        meta1.update({
            'width': window1.width,
            'height': window1.height,
            'transform': rasterio.windows.transform(window1, ref_src.transform)
        })
        
        meta2 = tgt_src.meta.copy()
        meta2.update({
            'width': window2.width,
            'height': window2.height,
            'transform': rasterio.windows.transform(window2, tgt_src.transform)
        })

        # Save the overlapping regions
        with rasterio.open(reference_out_path, 'w', **meta1) as ref_out:
            ref_out.write(ref_data)
        
        with rasterio.open(target_out_path, 'w', **meta2) as tgt_out:
            tgt_out.write(tgt_data)


def extract_date_from_filename(filename):
    """Extract date from filename using regex."""
    match = re.search(r"(\d{8})", filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d')
    return None

def main():
    # Get the paths from the wrapper function
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_directory", required = True, help = "Reference directory of the HLS data")
    parser.add_argument("--target_directory", required = True, help = "Target directory of the Landsat_ARD data")
    parser.add_argument("--processed_HLS", required = True, help = "Output directory for the processed HLS data")
    parser.add_argument("--processed_Landsat", required = True, help = "Output directory for the processed Landsat data")
    args = parser.parse_args()

    # reference_directory =  r"D:\SetoLab\HLS\Pre_Fusion_Crop_exclude_S30"
    reference_directory = args.reference_directory
    # target_directory = r"D:\SetoLab\Landsat_ARD\Pre_Fusion_Crop"
    target_directory = args.target_directory
    # output_directory_reference = "D:\SetoLab\HLS\Overlapped_Landsat"
    output_directory_reference = args.processed_HLS
    # output_directory_target = "D:\SetoLab\Landsat_ARD\Overlapped_SR"
    output_directory_target = args.processed_Landsat

    # Ensure output directories exist
    os.makedirs(output_directory_reference, exist_ok=True)
    os.makedirs(output_directory_target, exist_ok=True)

    #Get a list of files
    reference_files = [os.path.join(reference_directory, f) for f in os.listdir(reference_directory) if f.lower().endswith('.tif')]
    target_files = [os.path.join(target_directory, f) for f in os.listdir(target_directory) if f.lower().endswith('.tif')]



    #Now we go through the files and extract the overlapping areas
    #We do this by matching the dates in the HLS Pre_Fusion_Crop folder with the dates in the Landsat ARD Pre_Fusion_Crop folder
    for ref_file in reference_files:
        ref_date = extract_date_from_filename(ref_file)
        if ref_date is None:
            print(f"Could not extract date from {ref_file}. Skipping...")
            continue

        for tgt_file in target_files:
            tgt_date = extract_date_from_filename(tgt_file)
            if tgt_date is None:
                print(f"Could not extract date from {tgt_file}. Skipping...")
                continue

            #So we have now successfully extracted the dates from both the reference and target file. 
            #Perform a check to see if the date difference is within 1 day
            if abs((ref_date - tgt_date).days) <= 1:
                #Create output names based on input dataset names
                ref_out_name = os.path.basename(ref_file).replace('cropped', 'overlapped')
                tgt_out_name = os.path.basename(tgt_file).replace('prefusion', 'overlapped')

                #Define the output paths
                ref_out_path = os.path.join(output_directory_reference, ref_out_name)
                tgt_out_path = os.path.join(output_directory_target, tgt_out_name)

                #Extract the overlapping area
                extract_overlap_from_datasets(ref_file, tgt_file, ref_out_path, tgt_out_path)
                print(f"Processed overlap for {ref_file} and {tgt_file} and saved as {ref_out_name} and {tgt_out_name}")


if __name__ == '__main__':
    main()