#The purpose of this file will be to run the K-means classified landsat data through Meta's SAM (Segment Anything Model) to identify the objects in the raster
#The output will be fed into the OBSUM processing pipeline

#Import the packages
import numpy as np 
import torch
import matplotlib.pyplot as plt
import cv2
import rasterio
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

#Define a function to show the segmented masks
def show_anns(anns):
    if len(anns) == 0:
        #Since we have nothing
        return
    
    #Sort the masks so that the largest is first
    sorted_anns = sorted(anns, key = (lambda x: x['area']), reverse = True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        #Select three random RGB values for each mask
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    ax.imshow(img)

#The following functions are from the OBSUM team: read_raster, write_raster, linear_pct_stretch, color_composite
def read_raster(raster_path):
    dataset = rasterio.open(raster_path)
    raster_profile = dataset.profile
    raster = dataset.read()

    #Order the dimensions to be "Row, Column, Band" instead of "Band, Row, Column"
    raster = np.transpose(raster, (1, 2, 0))
    raster = raster.astype(np.dtype(raster_profile["dtype"]))

    return raster, raster_profile

def write_raster(raster, raster_profile, raster_path):
    raster_profile["dtype"] = str(raster.dtype)
    raster_profile["height"] = raster.shape[0]
    raster_profile["width"] = raster.shape[1]
    raster_profile["count"] = raster.shape[2]

    #Now we reorder the bands back to "Band, Row, Column"
    image = np.transpose(raster, (2, 0, 1))
    dataset = rasterio.open(raster_path, mode = 'w', **raster_profile)
    dataset.write(image)
    dataset.close()

def linear_pct_stretch(img, pct = 2, max_out = 1, min_out = 1):
    #This function calculates the lower and upper percentiles based on pct
    #Linearly rescales all the values between those percentiles to the min_out - max_out range
    #Clips values below min_out and max_out to avoid affecting the range
    #The pixel values are now stretched to improve contrast
    def gray_process(gray):
        truncated_down = np.percentile(gray, pct)
        truncated_up = np.percentile(gray, 100 - pct)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out

        return gray
    
    #The function now loops through all the bands, applies the gray process and then stacks them back into the same shape as the original 
    bands = []
    for band_idx in range(img.shape[2]):
        band = img[:, :, band_idx]
        band_strch = gray_process(band)
        bands.append(band_strch)
    
    img_pct_strch = np.stack(bands, axis = 2)
    return img_pct_strch

def color_composite(image, band_idx):
    #bands_idx are a list of band indices to select
    #Gets the band images we need and then stacks them by band
    image = np.stack([image[:, :, 1] for i in band_idx], axis = 2)
    return image


#Create a function that runs the landsat images through SAM in order to parallelize this
def create_SAM_masks(input_path, output_dir, sam):
    #Set up the SAM mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model = sam,
        points_per_side = 64,
        stability_score_thresh = 0.92,
        crop_n_layers = 1,
        crop_n_points_downscale_factor = 2,
        min_mask_region_area = 30
    )

    #Get the image and the meta data of the raster
    with rasterio.open(input_path) as src:
        image, profile = read_raster(input_path)
    
    #Normalize the raster for processing
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    #Convert the single-band grayscale into a 3-band
    image_rgb = np.repeat(image, 3, axis = 2)

    #Generate the masks using SAM
    masks = mask_generator.generate(image_rgb)

    #Sort the masks by area with the largest first and create an empty "objects" array
    masks = sorted(masks, key = (lambda x: x['area']), reverse = True)
    objects = np.full(shape = image.shape, fill_value = -1, dtype = np.int32)

    #Now we iterate through the masks and each object is assigned a unique label in the objects array along with recording the area of the mask
    for object_idx in range(len(masks)):
        mask = masks[object_idx]["segmentation"]
        objects[mask] = object_idx
        area = masks[object_idx]["area"]

    #Set the background pixels to -1 and expand the objects array to match the expected 3D shape for writing as a raster with 1 band
    objects[objects == -1] = len(masks)
    objects = np.expand_dims(objects, axis = 2)
    objects = objects.squeeze(-1)

    output_path = os.path.join(output_dir, os.path.basename(input_path).replace(".tif", "_SAM_mask.tif"))
    write_raster(objects, profile, output_path)

    return(output_path)


def main():
    #Let's parse in the arguments from the wrapper function
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", required = True, help = "The input directory of the Kmeans processed landsat files")
    parser.add_argument("--output_directory", required = True, help = "The output directory of the segmented files")
    parser.add_argument("--model_path", required = True, help = "The model path to the SAM model checkpoint")
    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory
    model_path = args.model_path

    #Ensure output directory exists
    os.makedirs(output_directory, exist_ok = True)

    #Load the SAM model once
    model_type = "vit_h"
    #Change to "cuda" if available
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint = model_path).to(device)

    #Get a list of all the files to process
    files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.lower().endswith('.tif')]

    #Parallelize with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers = 7) as executor:
        results = list(executor.map(create_SAM_masks, files, [output_directory] * len(files), [sam]*len(files)))

    for r in results:
        print(f"Finished creating masks for {r}")


if __name__ == "__main__":
    main()

