# %%
import pandas as pd
from tkinter import Tk, filedialog, simpledialog
import rainflow_analysis as rfa
import os
import time

# Hide the main Tkinter window
root = Tk()
root.withdraw()

####################################################################
# Determine the files of interest based on user input
####################################################################

# Prompt the user to enter a key string
key_string = simpledialog.askstring("Line Selection", "Enter the line name to search for (e.g., KS12, KS13, or KS14):")

if not key_string:
    print("No key string entered. Exiting.")
    exit()
else:
    print(f"Searching for line: {key_string}")
# %%
####################################################################
# Determine the features of interest
####################################################################

# Prompt the user to select an Excel file
excel_path = filedialog.askopenfilename(
    title="Select Pipe Tally File",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

if not excel_path:
    print("No file selected. Exiting.")
    exit()
else:
    print(f"Selected file: {excel_path}")

# Load the "Pipe Tally" sheet into a DataFrame
try:
    df_features = pd.read_excel(excel_path, sheet_name="Pipe Tally", header=0, engine='calamine')
    print("Extracted data from 'Pipe Tally' sheet.")
except Exception as e:
    print(f"Error loading 'Pipe Tally' sheet: {e}")

# Filter the DataFrame to include only rows where 'Feature Sub Type' is 'Dent'
df_dents = df_features[df_features['Feature Sub Type'] == 'Dent']
# %%
####################################################################
# Collect the pump/valve station metadata
####################################################################

# Prompt the user to select an Excel file containing pump/valve station metadata
stations_path = filedialog.askopenfilename(
    title="Select Pump Stations File",
    filetypes=[("Excel files",  "*.xlsx *.xls")]
)

if not stations_path:
    print("No metadata file selected. Exiting.")
    exit()

# Load the metadata file into a DataFrame
try:
    df_stations = pd.read_excel(stations_path, sheet_name=key_string, header=0)
    print(f"Extracted data from {key_string} sheet.")
except Exception as e:
    print(f"Error loading {key_string} sheet: {e}")
# %%
####################################################################
# Load the pressure history CSV file
####################################################################

pressure_history_path = filedialog.askopenfilename(
    title="Select Pressure History File",
    filetypes=[("CSV files", "*.csv")]
)

if not pressure_history_path:
    print("No pressure history file selected. Exiting.")
    exit()

# Load the pressure history CSV file into a DataFrame
try:
    df_pressure_history = pd.read_csv(pressure_history_path, header=0)
    print("Extracted data from pressure history CSV file.")
except Exception as e:
    print(f"Error loading pressure history CSV file: {e}")
# %%
####################################################################
# Perform RLA on each feature of interest
####################################################################

# Iterate through df_dents, and based on the value in column 'Absolute Distance [m]',
# find the upstream and downstream stations in df_stations based on column 'Continuous Measure (m)'
# with the station name in column 'Tag Name'. Then, use the upstream and downstream stations to select
# the pressure history data from df_pressure_history (the columns should match the Tag Names of the stations).

results = []

results_folder = 'KS13 Rainflow Results'
results_folder = os.path.join(os.getcwd(), results_folder) + '\\'
os.mkdir(results_folder)
count = 0

import importlib
importlib.reload(rfa)

for idx, dent in df_dents.iterrows():
    count += 1
    starttime = time.time()
    try:
        abs_dist = dent['Absolute Distance [ft]'] * 0.3048  # Convert feet to meters for comparison
        
        # Find upstream station
        upstream_candidates = df_stations[df_stations['Continuous Measure (m)'] <= abs_dist]
        
        # Find downstream station
        downstream_candidates = df_stations[df_stations['Continuous Measure (m)'] >= abs_dist]

        # Check if the pressure history contains the required columns
        if upstream_candidates.empty or downstream_candidates.empty:
            print(f"Could not find stations for Feature {dent['Feature ID']} at {abs_dist} m.")
            continue

        upstream_station = upstream_candidates.iloc[-1]
        downstream_station = downstream_candidates.iloc[0]
        upstream_tag = upstream_station['Tag Name']
        downstream_tag = downstream_station['Tag Name']
        
        # Select pressure history data for the upstream and downstream stations, and convert from kPa to psig
        upstream_pressure = pd.to_numeric(df_pressure_history[upstream_tag], errors='coerce').astype(float).to_numpy() * 0.145038
        downstream_pressure = pd.to_numeric(df_pressure_history[downstream_tag], errors='coerce').astype(float).to_numpy() * 0.145038
        time_data = pd.to_datetime(df_pressure_history['Date-Time']).to_numpy()
        
        # Build the dent dict. Convert to Imperial units if necessary.
        dd = rfa.DentData(
            dent_ID = dent['Feature ID'],
            OD = dent['TCPL NPS'], # Already using inch
            WT = dent['TCPL Nominal Wall Thickness [in]'],  # Already using inch
            SMYS = dent['TCPL SMYS [ksi]'] * 1000,  # Convert ksi to psi
            MAOP = dent['TCPL MOP/MAOP [psi]'],  # Already using psi
            service_years = (time_data[-1] - time_data[0]) / pd.Timedelta(days=365.25),  # Calculate service years from time data
            M = 3,  # Assume exponent of M = 3.0
            min_range = 0.0,  # Do not filter any psig values
            Lx = dent['Absolute Distance [ft]'],
            hx = dent['Elevation [ft]'],
            SG = 0.84, # Assumption from https://www.engineeringtoolbox.com/specific-gravity-liquid-fluids-d_294.html
            L1 = upstream_station['Continuous Measure (m)'] * 3.28084,  # Convert meters to feet
            L2 = downstream_station['Continuous Measure (m)'] * 3.28084,  # Convert meters to feet
            h1 = upstream_station['Elevation'] * 3.28084,  # Convert meters to feet
            h2 = downstream_station['Elevation'] * 3.28084,  # Convert meters to feet
            D1 = dent['TCPL NPS'],
            D2 = dent['TCPL NPS'],
        )
        # Determine the results path
        results_path = os.path.join(results_folder, f"Feature {dent['Feature ID']}") + '\\'
        os.mkdir(results_path)

        # Perform RLA for liquids
        # print(f"{count:04d} / {df_dents.shape[0]:04d} ({round(time.time() - starttime)}s): Processing Feature {dent['Feature ID']}...")
        SSI, CI, MD49_SSI, *_ = rfa.liquid([upstream_pressure, downstream_pressure], time_data, results_path, dd)
        
        result_dict = {
            'Feature ID': dd.dent_ID,
            'Upstream Station': upstream_tag,
            'Downstream Station': downstream_tag,
            'SSI': float(SSI).__round__(2),
            'CI': float(CI).__round__(2),
            'MD49_SSI': float(MD49_SSI).__round__(2)
        }
        results.append(result_dict)

        # Save results to notepad after each iteration
        with open(os.path.join(results_folder, "RLA_Results_Notepad.txt"), "a") as f:
            f.write(f"{count:04d} / {df_dents.shape[0]:04d} : {result_dict}\n")

        # Print or save the results as needed
        print(f"{count:04d} / {df_dents.shape[0]:04d} ({round(time.time() - starttime)}s): Finished processing Feature {dent['Feature ID']}")

    except Exception as e:
        print(f"{count:04d} / {df_dents.shape[0]:04d} ({round(time.time() - starttime)}s): Error processing Feature {dent['Feature ID']}: {e}")
        continue

# Save the df_results DataFrame to an Excel file
df_results = pd.DataFrame(results)
df_results.to_csv(results_path + 'RLA_Results.csv', index=False)
# %%
