"""
Python script that uses processing module that performs the following:
- Load the Pipe Tally containing all feature metadata
- Iterate through the .csv/.xlsx documents in a folder, searching for matching feature names
- Load the feature data and perform data smoothing using processing.smooth_data
- Export the smoothed data (and metadata) to the existing results Excel document for each feature saved in another directory
"""

import time
import os
import traceback
from tkinter import Tk, filedialog

import API1183_v2 as api

def main():
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()

    # Keep track of overall time
    global overall_start_time
    overall_start_time = time.time()

    # Split the workflow into three categories
    categories = ['KS12', 'KS13', 'KS14']

    ####################################################################
    # Select Results Output Excel Documents
    ####################################################################

    # Prompt the user to select the output folder containing the output result Excel documents
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if not output_folder:
        print("No folder selected. Exiting script.")
        exit()
    else:
        print(f"Selected folder: {output_folder}")

    # Check if the output data folder contains any sub folders for the categories
    output_subfolders = [f.path for f in os.scandir(output_folder) if f.is_dir() and any(cat in f.name for cat in categories)]
    if not output_subfolders:
        # No output_subfolders found, check for files directly in the output_folder
        print("No subfolders found for the specified categories, proceed with files in the main folder.")
        xlsm_files = [f.path for f in os.scandir(output_folder) if f.is_file() and f.name.endswith(('Results.xlsm'))]
        if not xlsm_files:
            print("No .xlsm files found. Exiting script.")
            exit()
        else:
            print(f"Found .xlsm files.")
    else:
        print(f"Found subfolders for output data.")
        # Iterate through each category
        for category in categories:
            # Select the corresponding output_subfolder if it contains a substring matching with the category
            output_subfolder = next((f for f in output_subfolders if category in f), None)
            if output_subfolder is None:
                print(f"No output subfolder found for category: {category}. Skipping.")
                continue
            # Begin processing the data for the current category. Only show the folder base name
            print(f"Begin processing for category: '{category}', using output folder: '{os.path.basename(output_subfolder)}'")
            # Using the selected output_subfolder, iterate through each sub folder, search for a file ending with 'Results.xlsm'
            xlsm_files = []
            for root, dirs, files in os.walk(output_subfolder):
                for file in files:
                    if file.endswith('Results.xlsm'):
                        xlsm_files.append(os.path.join(root, file))

            # Run the API1183 method for each xlsm file
            for xlsm_file in xlsm_files:
                start_time = time.time()
                feature_id = os.path.basename(xlsm_file).split("Feature_")[-1].split("_Results")[0]
                # Run the API1183 method
                try:
                    api.process_dent_file(xlsm_file)
                    print(f"{category} Feature {feature_id}: ({round(time.time() - start_time)}s/{round(time.time() - overall_start_time)}s) Successfully processed MD49: '{os.path.basename(xlsm_file)}'")
                    # Save results to notepad after each iteration. Keep the notepad in the parent output folder (one directory higher)
                    with open(os.path.join(os.path.dirname(output_subfolder), f"{category}_MD49_Results.txt"), "a") as f:
                        f.write(f"{category} Feature {feature_id}: ({round(time.time() - start_time)}s/{round(time.time() - overall_start_time)}s) Successfully processed MD49: '{os.path.basename(xlsm_file)}'\n")

                except Exception as e:
                    print(f"{category} Feature {feature_id}: ({round(time.time() - start_time)}s/{round(time.time() - overall_start_time)}s) Error processing MD49: {e}")
                    print(traceback.format_exc())
                    with open(os.path.join(os.path.dirname(output_subfolder), f"{category}_MD49_Results.txt"), "a") as f:
                        f.write(f"{category} Feature {feature_id}: ({round(time.time() - start_time)}s/{round(time.time() - overall_start_time)}s) Error processing MD49: '{os.path.basename(xlsm_file)}', {e}\n")
                        f.write(traceback.format_exc() + "\n")

# Run script when loaded
if __name__ == "__main__":
    main()
