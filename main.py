# %%
import pandas as pd
import rainflow_analysis as rfa
import os
import time

categories = ['KS12', 'KS13', 'KS14']
pipe_tally = r"C:\Users\emman\OneDrive - Softnostics\Projects\100001 - 100025\100004 (Acuren - Southbow Screening Software)\Client Documents\462-J209370\KS12-14 Data Collection - Copy.xlsx"
pump_stations = r"C:\Users\emman\OneDrive - Softnostics\Projects\100001 - 100025\100004 (Acuren - Southbow Screening Software)\Client Documents\462-J209370\Pump Stations.xlsx"

df_PT = pd.read_excel(pipe_tally, sheet_name="Sheet1", header=1, engine='calamine')

for category in categories:
    print(f"Processing category: {category}")
    # Filter the DataFrame for the current category
    df_category = df_PT[df_PT['Section'].astype(str).str.contains(category, case=False, na=False)]
    df_dents = df_category[df_category['Feature Type'] == 'Dent']

    # Load the Pump Stations file
    df_stations = pd.read_excel(pump_stations, sheet_name=category, header=0, engine='calamine')

    # Load the pressure history CSV file in folder "Combined Data" based on matching substring
    pressure_history_path = os.path.join(os.getcwd(), "Combined Data", f"{category}_combined.csv")
    df_pressure_history = pd.read_csv(pressure_history_path, header=0)

    ####################################################################
    # Perform RLA on each feature of interest
    ####################################################################

    # Iterate through df_dents, and based on the value in column 'AP Measure (m)',
    # find the upstream and downstream stations in df_stations based on column 'Continuous Measure (m)'
    # with the station name in column 'Tag Name'. Then, use the upstream and downstream stations to select
    # the pressure history data from df_pressure_history (the columns should match the Tag Names of the stations).

    results = []

    results_folder = f"{category} Rainflow Results"
    results_folder = os.path.join(os.getcwd(), results_folder) + '\\'
    os.mkdir(results_folder)
    count = 0

    import importlib
    importlib.reload(rfa)

    for idx, dent in df_dents.iterrows():
        count += 1
        starttime = time.time()
        try:
            abs_dist = dent['AP Measure (m)']
            
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
                WT = dent['TCPL Nominal Wall Thickness [mm]'] * 0.0393701,  # Convert mm to inch
                SMYS = dent['TCPL SMYS [MPa]'] * 145.038,  # Convert MPa to psi
                MAOP = dent['TCPL MOP/MAOP [kPa]'] * 0.145038,  # Convert kPa to psi
                service_years = (time_data[-1] - time_data[0]) / pd.Timedelta(days=365.25),  # Calculate service years from time data
                M = 3,  # Assume exponent of M = 3.0
                min_range = 0.0,  # Do not filter any psig values
                Lx = dent['AP Measure (m)'] * 3.28084,  # Convert meters to feet
                hx = dent['Dent Elevation [m]'] * 3.28084,  # Convert meters to feet
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
                'Category': category,
                'Feature ID': dd.dent_ID,
                'Upstream Station': upstream_tag,
                'Downstream Station': downstream_tag,
                'SSI': float(SSI).__round__(2),
                'CI': float(CI).__round__(2),
                'MD49_SSI': float(MD49_SSI).__round__(2)
            }
            results.append(result_dict)

            # Save results to notepad after each iteration
            with open(os.path.join(results_folder, f"{category}_RLA_Results.txt"), "a") as f:
                f.write(f"{result_dict}\n")

            # Print or save the results as needed
            print(f"{count:04d} / {df_dents.shape[0]:04d} ({round(time.time() - starttime)}s): Finished processing {category} Feature {dent['Feature ID']}")

        except Exception as e:
            print(f"{count:04d} / {df_dents.shape[0]:04d} ({round(time.time() - starttime)}s): Error processing {category} Feature {dent['Feature ID']}: {e}")
            continue

    # Save the df_results DataFrame to an Excel file
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_path + f"{category}_RLA_Results.csv", index=False)

# %%
