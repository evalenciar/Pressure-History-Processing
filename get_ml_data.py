'''
The goal of this script is to merge data from two different Excel files and adding new columns of data to the Master file.

This Python script loads two tables: Master and Reference pipe tally files. The Master file contains all of the features of interest, while the Reference file contains
additional data not found in the Master file. The common denominator between the two files is the "Feature Type", "Feature Sub Type", and "Feature ID" columns.
The Reference file contains a column "Interacting Feature Type" and "Interacting Feature ID" which indicates the Metal Loss type and feature ID, respectively, that is interacting with the feature in that row.
The goal is to iterate through each feature in the Master file, find the corresponding feature (based on Feature Type, Feature Sub Type, and Feature ID) in the Reference file,
if there is "Metal Loss" in "Interacting Feature Type" then extract the "Interacting Feature ID" and search for that new Feature ID in the Reference file (using the Feature Type = Metal Loss and Feature ID = Interacting Feature ID).
Then, extract the "Peak Depth [%]" and "Wall Surface" columns from the Reference file and add them to the Master file as new columns "Interacting Peak Depth [%]" and "Interacting Wall Surface".
'''
# %%
import pandas as pd
import numpy as np 
import os
import re
import warnings

def load_excel_data(file_path, sheet_name="Sheet1", header=1, engine='calamine'):
    """
    Load data from an Excel file into a pandas DataFrame.
    
    :param file_path: Path to the Excel file.
    :param sheet_name: Name or index of the sheet to load (default is the first sheet).
    :return: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header, engine=engine)
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
def merge_data(master_df, reference_df, line_segment):
    """
    Merge data from the reference DataFrame into the master DataFrame based on specified criteria.
    
    :param master_df: DataFrame containing the master data.
    :param reference_df: DataFrame containing the reference data.
    :return: Merged DataFrame with additional columns from the reference data.
    """

    print(f"Processing line segment: {line_segment}")
    # Filter the DataFrame for the current category
    df_category = master_df[master_df['Section'].astype(str).str.contains(line_segment, case=False, na=False)]
    df_dents = df_category[df_category['Feature Type'] == 'Dent']
    
    # Iterate through each row in the master DataFrame
    for idx, master_row in df_dents.iterrows():
        feature_id = master_row['Feature ID']
        feature_type = master_row['Feature Type']
        feature_sub_type = master_row['Feature Sub Type']
        
        # Find corresponding rows in the reference DataFrame
        ref_rows = reference_df[
            (reference_df['Feature ID'] == feature_id)
            & (reference_df['Feature Sub Type'] == feature_type)
            & (reference_df['Feature Type'] == feature_sub_type)
        ]
        
        if not ref_rows.empty:
            interacting_feature_id = ref_rows.iloc[0]['Interacting Feature ID']
            interacting_feature_type = ref_rows.iloc[0]['Interacting Feature Type']
            interacting_feature_sub_type = ref_rows.iloc[0]['Interacting Feature Sub Type']
            
            if interacting_feature_type == 'Metal Loss':
                # Find the interacting feature in the reference DataFrame
                interacting_rows = reference_df[
                    (reference_df['Feature ID'] == interacting_feature_id) &
                    (reference_df['Feature Type'] == 'Metal Loss') &
                    (reference_df['Feature Sub Type'] == interacting_feature_sub_type)
                ]
                
                if not interacting_rows.empty:
                    # First confirm that Interacting Feature ID, Interacting Feature Type, and Interacting Feature Sub Type
                    # matches the original dent feature ID. If any do not match, then skip that row and print a warning.

                    if np.any([interacting_rows.iloc[0]['Interacting Feature ID'].astype(int) != feature_id,
                                interacting_rows.iloc[0]['Interacting Feature Sub Type'] != feature_type,
                                interacting_rows.iloc[0]['Interacting Feature Type'] != feature_sub_type]):
                        print(f"Warning: Metal Loss {interacting_feature_sub_type} ID {interacting_feature_id} has Interacting Feature ID {interacting_rows.iloc[0]['Interacting Feature ID']} that does not match original dent Feature ID {feature_id} for row {idx}")
                        continue

                    # Extract the required data and add it to the master DataFrame
                    peak_depth = interacting_rows.iloc[0]['Peak Depth [%]']
                    wall_surface = interacting_rows.iloc[0]['Wall Surface']
                    
                    master_df.at[idx, 'Interacting Peak Depth [%]'] = peak_depth
                    master_df.at[idx, 'Interacting Wall Surface'] = wall_surface
                    master_df.at[idx, 'Interacting Feature ID'] = int(interacting_feature_id)
                    master_df.at[idx, 'Interacting Feature Type'] = interacting_feature_sub_type
                    master_df.at[idx, 'Interacting Feature Sub Type'] = interacting_feature_type
                    print(f"Info: Updated Feature ID {feature_id} (row {idx}) with Interacting Feature ID {interacting_feature_id}, Peak Depth {peak_depth}, Wall Surface {wall_surface}")
                else:
                    print(f"Warning: Feature ID {feature_id} (row {idx}) has Interacting Feature ID {interacting_feature_id} which does not exist in reference data.")
            # Check if Interacting Feature Type is NaN and handle it
            elif pd.isna(interacting_feature_type):
                # print(f"Info: Interacting Feature Type is NaN for Feature ID {feature_id} in row {idx}. No interacting feature to process.")
                continue
            else:
                print(f"Warning: Interacting Feature Type is not 'Metal Loss' for Feature ID {feature_id} in row {idx}")
                continue
        else:
            # print(f"Warning: No matching feature found in reference data for Feature ID {feature_id} in row {idx}")
            continue
    return master_df

def save_to_excel(df, output_path):
    """
    Save the DataFrame to an Excel file.
    
    :param df: DataFrame to save.
    :param output_path: Path to the output Excel file.
    """
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving to Excel file: {e}")

def main():
    # File paths
    master_file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Pipe Tally\KS12-14 Data Collection - (Fixed Headers).xlsx"
    ks12_reference_file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Pipe Tally\KS12 201535_Final_Final Pipe Tally - Metric_Rev_2_MFL-AXIAL_HR_GEO_HR.xlsx"
    ks13_reference_file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Pipe Tally\KS13 201536_Final_Final Pipe Tally - Imperial_Rev_2_MFL-AXIAL_HR_GEO_HR.xlsx"
    ks14_reference_file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Pipe Tally\KS14 201537_Final_Final Pipe Tally - Imperial_Rev_1_MFL-AXIAL_HR_GEO_HR.xlsx"
    output_file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Pipe Tally\KS12-14 Data Collection - (Fixed Headers) Merged.xlsx"
    
    # Load data
    master_df = load_excel_data(master_file_path)
    ks12_reference_df = load_excel_data(ks12_reference_file_path, sheet_name="Pipe Tally", header=0)
    ks13_reference_df = load_excel_data(ks13_reference_file_path, sheet_name="Pipe Tally", header=0)
    ks14_reference_df = load_excel_data(ks14_reference_file_path, sheet_name="Pipe Tally", header=0)

    if master_df is None or ks12_reference_df is None or ks13_reference_df is None or ks14_reference_df is None:
        print("Failed to load data. Exiting.")
        return
    else:
        print("Data loaded successfully.")
    
    # Define line segments to process
    line_segments = ['KS12', 'KS13', 'KS14'] 

    # Create new columns in the master DataFrame for the additional data. Use dtype object to allow for string values
    master_df['Interacting Peak Depth [%]'] = pd.Series([np.nan]*len(master_df), dtype=object)
    master_df['Interacting Wall Surface'] = pd.Series([np.nan]*len(master_df), dtype=object)
    master_df['Interacting Feature ID'] = pd.Series([np.nan]*len(master_df), dtype=object)
    master_df['Interacting Feature Type'] = pd.Series([np.nan]*len(master_df), dtype=object)
    master_df['Interacting Feature Sub Type'] = pd.Series([np.nan]*len(master_df), dtype=object)
        
    # Process each line segment and merge data
    for line_segment in line_segments:
        master_df = merge_data(master_df, ks12_reference_df if line_segment == 'KS12' else ks13_reference_df if line_segment == 'KS13' else ks14_reference_df, line_segment)
    
    # Save the merged data to a new Excel file
    save_to_excel(master_df, output_file_path)
# %%
if __name__ == "__main__":
    main()