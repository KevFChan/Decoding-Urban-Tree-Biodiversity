#Import the libraries
import os
import tarfile
import concurrent.futures
import argparse


#Parse in the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_directory", required = True, help = f"Input directory of the .TAR Landsat ARD files")
parser.add_argument("--output_directory", required = True, help = f"Output directory of the extracted Landsat ARD files")
parser.add_argument("--city", required = True, help = f"City")
parse_args = parser.parse_args()

#Define the cities
city = parse_args.city

#Now we loop through the directory containing the tar files and we extract them
#for city in cities:
#Replace with wrapper function input
#"/Volumes/Seagate Portable Drive/Central_Park_Climate/Sample_data/Landsat_ARD_tar"
source_directory = parse_args.input_directory

target_directories = {
    '_SR':'/SR'
}

#Define the output directory
output_directory = parse_args.output_directory

#Create target directories here if they do not exist
for target_directory in target_directories.values():
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def extract_tarfile(filename):
    """
    Extracts a given TAR file to the appropriate directory based on its ending.
    """
    for ending, target_directory in target_directories.items():
        if ending in filename:
            with tarfile.open(os.path.join(source_directory, filename), 'r') as archive:
                archive.extractall(output_directory)
            print(f"Extracted {filename} to {output_directory}")
            break

#Extract the files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
    #List of filenames to process
    filenames = [filename for filename in os.listdir(source_directory) if filename.endswith(".tar")]

    #Extract each file
    futures = [executor.submit(extract_tarfile, filename) for filename in filenames]

    #Wait for all futures to complete
    concurrent.futures.wait(futures)

print("Files have been extracted to their respective repositories")

