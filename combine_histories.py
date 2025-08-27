# %%
import os
import pandas as pd
import collections
from tkinter import Tk, filedialog, simpledialog
import time

# Prompt the user to select a folder
folder_path = filedialog.askdirectory(title='Select Folder Containing Files')

# Get all file names in the folder
all_files = os.listdir(folder_path)

# After matching_files is created and saved, process Excel files in matching_files
excel_files = [os.path.join(folder_path, f) for f in all_files if f.lower().endswith(('.xlsx', '.xls'))]

# %%
# Group files by their column labels
column_groups = collections.defaultdict(list)

for file in excel_files:
    try:
        starttime = time.time()
        df = pd.read_excel(file, sheet_name="Samples", engine='calamine')
        # Use tuple of column names as the group key
        col_key = tuple(df.columns)
        column_groups[col_key].append((file, df))
        print(f"Loaded {round(time.time() - starttime)}s: {file[-20:]}")
    except Exception as e:
        print(f"Error loading '{file}': {e}")
# %%
# Combine DataFrames for each group and save them
combined_dfs = {}
for idx, (col_key, file_df_list) in enumerate(column_groups.items(), start=1):
    dfs = []
    for file, df in file_df_list:
        # Set the first column as index (assumed datetime)
        df.set_index(df.columns[0], inplace=True)
        dfs.append(df)
    combined_df = pd.concat(dfs)
    set_name = f"SET{idx}"
    combined_dfs[set_name] = combined_df
    print(f"Combined DataFrame for {set_name}:")
    print(combined_df)
    # Optionally, save each combined DataFrame to Excel

original_combined_dfs = combined_dfs.copy()

# %%
### Specifically for KS12 pressure history processing

# # Check out SET3
# df = combined_dfs['SET3']

# # Filter TPELO-00A-B0-PMLS: keep only rows before or at 2024-03-05 07:00
# mask_pmls = df.index >= pd.Timestamp('2024-03-05 07:12')
# df.loc[mask_pmls, ['TPELO-00A-B0-PMLS']] = None

# # Filter TPELO-B0-PMLD: keep only rows after or at 2024-06-13 13:00
# mask_pmld = df.index <= pd.Timestamp('2024-06-13 13:00')
# df.loc[mask_pmld, ['TPELO-B0-PMLD']] = None

# # Merge the two columns into a single Series, preferring non-null values
# df['TPELO-B0-PMLD'] = df['TPELO-B0-PMLD'].combine_first(df['TPELO-00A-B0-PMLS'])

# # Drop the unused column
# df.drop(columns=['TPELO-00A-B0-PMLS'], inplace=True)

# # Update the combined DataFrame in the dictionary
# combined_dfs['SET3'] = df

# # Check out SET1
# df = combined_dfs['SET1']

# # Filter TPELO-00A-B0-PMLS: keep only rows before or at 2024-03-05 07:00
# mask_pmls = df.index >= pd.Timestamp('2024-03-05 07:12')
# df.loc[mask_pmls, ['TPELO-00A-B0-PMLS']] = None

# # Filter TPELO-B0-PMLD: keep only rows after or at 2024-06-13 13:00
# mask_pmld = df.index <= pd.Timestamp('2024-06-13 13:00')
# df.loc[mask_pmld, ['TPELO-B0-PMLS']] = None

# # Merge the two columns into a single Series, preferring non-null values
# df['TPELO-B0-PMLS'] = df['TPELO-B0-PMLS'].combine_first(df['TPELO-00A-B0-PMLS'])

# # Drop the unused column
# df.drop(columns=['TPELO-00A-B0-PMLS'], inplace=True)

# # Update the combined DataFrame in the dictionary
# combined_dfs['SET1'] = df

# %%

# Combine all DataFrames into a single DataFrame
final_combined_df = combined_dfs['SET1'].join([combined_dfs['SET2'],combined_dfs['SET3'],combined_dfs['SET4']])
# %%
final_combined_df.to_csv('KS13_combined-1.csv')
# %%
