#This .py file will execute the preprocessing and processing scripts on the remote sensing data for a city using OBSUM for the spatiotemporal fusion step
#We will be able to define the input and output files of each processing step here as well
import subprocess
import os
import argparse

#Define the function we will use to run the scripts
def run_script(script, working_dir, args = ""):
    try:
        #Change the working directory
        os.chdir(working_dir)
        print(f"Current working directory is set to: {os.getcwd()}")

        #Construct the arguments list
        command = ['python', script] + args
        print(f"Running command: {' '.join(command)}")

        #Run the script as a subprocess
        subprocess.run(command, check = True)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        exit(1)

#Now we will run the scripts in succession
if __name__ == "__main__":
    #Define the working directory of the scripts based on the machine this code is running on
    #Define the path we want the data to be processed to based on the machine this code is running on
    script_working_dir = "/Volumes/Seagate Portable Drive/Central_Park_Climate/Streamlined_Scripts_City"
    data_input_dir = "/Volumes/Seagate Portable Drive/Central_Park_Climate/Sample_data"
    data_output_dir = "/Volumes/Seagate Portable Drive/Central_Park_Climate/Streamlined_Outputs"
    

    #Check first that the data output cirectory exists
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)

    #Define the list of cities we want to process the remote sensing data for:
    #cities = ["Boston", "Tampa", "New York City", "Los Angeles"]
    cities = ["New_York_City"]
    #Define the longitudes and latitudes of the cities 
    # longitudes = [-71.057083, -82.452606, -73.935242, -118.243683]
    # latitudes = [42.361145, 27.964157, 40.730610, 34.052235]
    longitudes = ["-73.935242"]
    latitudes = ["40.730610"]
    #Initialize whether or not a city is on the intersection of two tiles
    boundary = [False]

    for num_city in range(len(cities)):
        #Define the city and its longitude and latitude
        city = cities[num_city]
        lon = longitudes[num_city]
        lat = latitudes[num_city]

        print(f"Starting to process remote sensing data for {city}")

        #Step 0.1: Run the Landsat extraction code
        step_zero_one_input = os.path.join(data_input_dir, "Landsat_ARD")
        step_zero_one_output = os.path.join(data_output_dir, "Step_Zero_One_Extracted_Landsat_ARD", city)
        print(f"Running the Landsat ARD extraction script on {city}'s data")
        # run_script("1_Landsat_ARD_extract_par_city.py", script_working_dir, ["--input_directory", step_zero_one_input,
        #                                                                      "--output_directory", step_zero_one_output, 
        #                                                                      "--city", city])
        
        #Step 0.2: Run the Landsat stacking code
        #Note! You may need to go into the file and change ".TIF" into ".tif"
        step_zero_two_input = step_zero_one_output
        step_zero_two_output = os.path.join(data_output_dir, "Step_Zero_Two_Stacked_Landsat_ARD", city)
        print(f"Running the Landsat ARD stacking script on {city}'s data")
        # run_script("2_Landsat_ARD_stacking_par_city.py", script_working_dir, ["--input_directory", step_zero_two_input,
        #                                                                       "--output_directory", step_zero_two_output])
        
        #Step 0.3: If the city is on a boundary, then we will run the merging stacked tile code for Landsat ARD
        if boundary[num_city] == True:
            step_zero_three_input = step_zero_two_output
            step_zero_three_output = os.path.join(data_output_dir, "Step_Zero_Three_Merged_Landsat_ARD", city)
            print(f"Running the Landsat ARD merging script on {city}'s data")
            # run_script("3_Landsat_ARD_merging_two_stacked_tile_par_city.py", script_working_dir, ["--input_directory", step_zero_three_input,
            #                                                                                       "--output_directory", step_zero_three_output])
            #Now we need to replace the step_zero_two_output path
            step_zero_two_output = step_zero_three_output


        #Step 1: Run the reprojection and band extraction script on the Landsat_ARD data
        #The script creates tif files in "Step_One_Reprojected_Extracted_SR_City" with the format of 'stacked_filename_city'
        #Define the input and output directories
        #step_one_input = os.path.join(r'C:\Temp\Landsat_ARD\Stacked_SR', city)
        step_one_input = step_zero_two_output
        step_one_output = os.path.join(data_output_dir, "Step_One_Reprojected_Extracted_SR", city)
        print(f"Running the reprojection and band extraction script on {city}'s SDC500m data")
        # run_script("pre_0_1_reprojection_band_extraction_LandsatARD_city.py", script_working_dir, ["--input_directory", step_one_input, 
        #                                                                                      "--output_directory", step_one_output])
        

        #Step 2: Run the stacking, reprojection, and resampling script of SDC500m
        #The script creates resampled data in "Step_Two_Resampled_Output" directory with subfolders separated by city
        #Define the input and output directories, we define the stack and reprojected directories as intermediate steps. This script will be deleting those
        #We will also be inputting the latitude and longitude of the city here
        #step_two_input = os.path.join(r'F:/SDC/', city)
        #Debug these lines of code:
        step_two_input = os.path.join(data_input_dir, "stacked_SDC")
        step_two_stacked = os.path.join(data_output_dir, "Step_Two_Stacked_Output", city)
        step_two_reprojected = os.path.join(data_output_dir, "Step_Two_Reprojected_Output", city)
        step_two_resampled = os.path.join(data_output_dir, "Step_Two_Resampled_Output", city)
        print(f"Running the stacking, reprojection, and resampling SDC500m data processing script for {city}")
        # run_script('pre_0_stack_reproject_resample_SDC500m_tiles_city.py', script_working_dir, ["--input_directory", step_two_input, "--stacked_directory", step_two_stacked, 
        #                                                                      "--reprojected_directory", step_two_reprojected, "--resampled_directory", step_two_resampled,
        #                                                                      "--longitude", lon, "--latitude", lat])
        
    
        #Step 3: Run the crop, fusion, and extend processes on the LandsatARD and MODIS data. One of the inputs should be the data from Step One
        step_three_reference = os.path.join(data_input_dir, "reference_file")
        #Use the output of step 1 as our input
        step_three_target_ARD = step_one_output
        step_three_output_ARD = os.path.join(data_output_dir, "Step_Three_Landsat_Pre_Fusion_Crop_Full", city)

        #Define the paths for MODIS/SDC
        step_three_target_MODIS = step_two_resampled
        step_three_output_MODIS = os.path.join(data_output_dir, "Step_Three_MODIS_Pre_Fusion_Crop", city)
        print(f"Running the crop, fusion, and extend processes on the LandsatARD and MODIS data for {city}")
        #Now Landsat and MODIS should have the same area
        # run_script('pro_0_crop_fusion_extend_LandsatARD_MODIS_par_city.py', script_working_dir, ["--reference_directory", step_three_reference, "--target_ARD_directory", step_three_target_ARD,
        #                                                                                          "--output_ARD_directory", step_three_output_ARD, "--target_MODIS_directory", step_three_target_MODIS,
        #                                                                                          "--output_MODIS_directory", step_three_output_MODIS, "--city", city])


        #Step 4: Run the overlapping area HLS Landsat script to extract the overlapping areas between the HLS data and the Landsat_ARD_data
        step_four_reference = os.path.join(data_input_dir, "HLS")
        #Use only one file/image
        step_four_target = step_three_output_MODIS
        step_four_reference_output = os.path.join(data_output_dir, "Step_Four_Processed_HLS", city)
        #"D:\SetoLab\HLS\Overlapped_Landsat"
        step_four_directory_output = os.path.join(data_output_dir, "Step_Four_Processed_Landsat", city)
        #"D:\SetoLab\Landsat_ARD\Overlapped_SR"
        print(f"Getting the overlapping areas between the HLS data and the Landsat data for {city}")
        # run_script('pre_3_Overlapping_area_HLS_Landsat_city.py', script_working_dir, ["--reference_directory", step_four_reference, "--target_directory", step_four_target,
        #                                                                            "--processed_HLS", step_four_reference_output, "--processed_Landsat", step_four_directory_output])

        #Step 4.5: Train the XGboost model on Landsat data
        #This should be in the loop, landsat ARD predicts HLS
        step_four_five_input_landsat = step_four_directory_output
        step_four_five_input_HLS = step_four_reference_output
        step_four_five_output = os.path.join(data_output_dir, "XGBoost_Models", city)
        print(f"Training the XGBoost models for {city} by predicting the HLS values using Landsat ARD data")
        #Band wise regression model
        #Output of Landsat ARD should be similar to HLS


        #Step 5: Apply XGBoost Regression on the ARD data after the crop_fusion_extend script finished running
        step_five_input = step_three_output_ARD
        #ARD and HLS use the same sensor 
        #Change with the new trained models
        step_five_models = os.path.join(data_input_dir, "XGBoost")
        step_five_output = os.path.join(data_output_dir, "Step_Five_Harmonized_Full", city)
        print(f"Applying XGBoost regression on the HLS data for {city} after crop_fusion_extend step has ran")
        # run_script('pre_5_Harmonization_model_apply_HLS_Landsat_par_city.py', script_working_dir, ["--input_directory", step_five_input, 
        #                                                                                            "--models", step_five_models,
        #                                                                                            "--output_directory", step_five_output])
        
        
        #Step 6: Determine which images are clear in the Landsat Harmonized files
        step_six_input = step_five_output
        #ATTENTION: Need to change the if statement in the .py file to only take in clear images
        step_six_output = os.path.join(data_output_dir, "Step_Six_Pre_Fusion_Clear_Full", city)
        print(f"Running the LandsatARD processing code to determine which images are clear for {city}")
        # run_script('pro_1_clear_LandsatARD_pre_image_fusion_par_city.py', script_working_dir, ["--input_directory", step_six_input,
        #                                                                                            "--output_directory", step_six_output])
        

        #Step 7: Run the Pre_Fusion data set through a Kmeans classifier
        #Training the Kmeans classifier
        step_seven_input = step_five_output
        step_seven_output = os.path.join(data_output_dir, "Step_Seven_Pre_Fusion_Class_full", city)
        #Output individual maps with classified pixels
        print(f"Running Pre-Fusion 500m data for {city} through Kmeans")
        # run_script('pro_2_classification_map_Kmeans_city.py', script_working_dir, ["--input_directory", step_seven_input,
        #                                                                            "--output_directory", step_seven_output])
        

        #Step 8: Upsample the SDC 500 m data 
        step_eight_input = step_three_output_MODIS
        step_eight_output = os.path.join(data_output_dir, "Step_Eight_Pre_Fusion_Upsample", city)
        print(f"Upsample the SDC500m data for {city}")
        # run_script('pro_3_upsampling_SDC500m_pre_image_fusion_par_city.py', script_working_dir, ["--input_directory", step_eight_input,
        #                                                                                          "--output_directory", step_eight_output])
        
        
        #Step 8.5: Run the Pre_Fusion data through Facebooks Segment-Anything-Model (SAM)
        step_eight_five_input = step_seven_output
        step_eight_five_output = os.path.join(data_output_dir, "Step_Eight_Five_Post_SAM_Processing", city)
        step_eight_five_model = "/Volumes/Seagate Portable Drive/Central_Park_Climate/Sample_data/SAM_Code/segment_anything/models/sam_vit_h_4b8939.pth"
        print(f"Run the Segment-Anything-Model from Meta on the landsat data")
        # run_script("pro_3_5_SAM_city.py", script_working_dir, ["--input_directory", step_eight_five_input,
        #                                                        "--output_directory", step_eight_five_output,
        #                                                        "--model_path", step_eight_five_model])


        #Step 9: Run image fusion on the Landsat_ARD and MODIS data and perform OBSUM
        #We want to fuse the 30 meter 16-day Landsat ARD with the 500 meter daily MODIS
        #Landsat is the fine image and MODIS is the coarse image
        step_nine_path_landsat = step_five_output
        step_nine_path_MODIS = step_three_output_MODIS
        step_nine_path_SAM = step_eight_five_output
        step_nine_path_class = step_seven_output
        step_nine_output = os.path.join(data_output_dir, "Step_Nine_Spatiotemporal_Fused_OBSUM", city)
        print(f"Running OBSUM image fusion code on MODIS and Landsat data for {city}")
        run_script('pro_4_OBSUM_image_fusion_par_SDC_full_city.py', script_working_dir, ["--path_landsat", step_nine_path_landsat,
                                                                                         "--path_MODIS", step_nine_path_MODIS,
                                                                                         "--path_SAM", step_nine_path_SAM,
                                                                                         "--path_class", step_nine_path_class,
                                                                                         "--output_directory", step_nine_output])


        step_nine_executable_path = os.path.join(data_input_dir, "cuFSDAF-master")
        step_nine_template_path = os.path.join(data_input_dir, "cuFSDAF-master/Code/Parameters.txt")
        step_nine_parameter_directory = os.path.join(data_output_dir, "Step_Nine_parameter_HLS_MODIS", city)


        print("Processing is finished!")
        

        


        

