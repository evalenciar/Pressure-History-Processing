
import pandas as pd
import numpy as np
import md_processing as mdp
import md49
import rainflow_analysis as rfa

file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Validation\Level 2 MD24 Validation Data.xlsx"

# Load the Excel file. Pipe information is in Pipe sheet. Cycles information is in Cycles sheet. And Profiles are in Profiles sheet.
xls = pd.ExcelFile(file_path)
pipe_df = pd.read_excel(xls, 'Pipe')
cycles_df = pd.read_excel(xls, 'Cycles', header=None)
rainflow_df = pd.read_excel(xls, 'Rainflow', header=None)
# profiles_df = pd.read_excel(xls, 'Profiles', index_col=0)

# Create a MD49 instance for processing
pf = md49.CreateProfiles(process_data=False)
pf._results_axial_us = {"lengths": list(pd.read_excel(xls, "LAX_US", index_col=0,).to_dict().values())[0], "areas": list(pd.read_excel(xls, "AAX_US", index_col=0).to_dict().values())[0]}
pf._results_axial_ds = {"lengths": list(pd.read_excel(xls, "LAX_DS", index_col=0).to_dict().values())[0], "areas": list(pd.read_excel(xls, "AAX_DS", index_col=0).to_dict().values())[0]}
pf._results_circ_us_ccw = {"lengths": list(pd.read_excel(xls, "LTR_US_CCW", index_col=0).to_dict().values())[0], "areas": list(pd.read_excel(xls, "ATR_US_CCW", index_col=0).to_dict().values())[0]}
pf._results_circ_us_cw = {"lengths": list(pd.read_excel(xls, "LTR_US_CW", index_col=0).to_dict().values())[0], "areas": list(pd.read_excel(xls, "ATR_US_CW", index_col=0).to_dict().values())[0]}
pf._results_circ_ds_ccw = {"lengths": list(pd.read_excel(xls, "LTR_DS_CCW", index_col=0).to_dict().values())[0], "areas": list(pd.read_excel(xls, "ATR_DS_CCW", index_col=0).to_dict().values())[0]}
pf._results_circ_ds_cw = {"lengths": list(pd.read_excel(xls, "LTR_DS_CW", index_col=0).to_dict().values())[0], "areas": list(pd.read_excel(xls, "ATR_DS_CW", index_col=0).to_dict().values())[0]}
# print(pf._results_axial_us)
# print(pf._results_axial_ds)
# print(pf._results_circ_us_ccw)
# print(pf._results_circ_us_cw)
# print(pf._results_circ_ds_ccw)
# print(pf._results_circ_ds_cw)


# Create a MD49 instance for processing
dd = rfa.DentData(
    dent_category = "TEST",
    dent_ID = "TEST ID",
    OD = 20,
    WT = 0.281,
    SMYS = 52000,
    MAOP = 0,
    service_years = 1.0,
    M = 3,  # Assume exponent of M = 3.0
    min_range = 5,  # Filter at 5psig
    Lx = 0,
    hx = 0,
    SG = 0.84, # Assumption from https://www.engineeringtoolbox.com/specific-gravity-liquid-fluids-d_294.html
    L1 = 0,
    L2 = 0,
    h1 = 0,
    h2 = 0,
    D1 = 0,
    D2 = 0,
    confidence = 80,
    CPS = True,
    interaction_weld = True,
    interaction_corrosion = True,
    dent_depth_percent = 1.280,
    ili_pressure = 789,
    restraint_condition = "Unrestrained",
    ml_depth_percent = 0,
    ml_location = "OD",
    orientation = "Orientation",
    vendor_comments = "FALSE"
)
md49_results = mdp.process(dd, pf, (7783.80608, 0, 0, rainflow_df.values, cycles_df.values), False)
print(md49_results)
# Save the results to a txt file
with open("validation_results.txt", "w") as f:
    f.write(str(md49_results))
