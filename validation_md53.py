
import pandas as pd
import numpy as np
import md_processing as mdp
import md49
import rainflow_analysis as rfa

file_path = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Validation\MD53 Validation Data.xlsx"

# Load the Excel file. Pipe information is in Pipe sheet. Cycles information is in Cycles sheet. And Profiles are in Profiles sheet.
xls = pd.ExcelFile(file_path)
pipe_df = pd.read_excel(xls, 'Pipe')
cycles_df = pd.read_excel(xls, 'Cycles', header=None)
# profiles_df = pd.read_excel(xls, 'Profiles', index_col=0)

# Create a MD49 instance for processing
pf = md49.CreateProfiles(process_data=False)
pf._results_axial_us = {"lengths": {key: {"length": val, "other": np.nan} for key, val in pd.read_excel(xls, "LAX_US", index_col=0).to_dict()["LAX_US"].items()}, "areas": pd.read_excel(xls, "AAX_US", index_col=0).to_dict()["AAX_US"]}
pf._results_axial_ds = {"lengths": {key: {"length": val, "other": np.nan} for key, val in pd.read_excel(xls, "LAX_DS", index_col=0).to_dict()["LAX_DS"].items()}, "areas": pd.read_excel(xls, "AAX_DS", index_col=0).to_dict()["AAX_DS"]}
pf._results_circ_us_ccw = {"lengths": {key: {"length": val, "other": np.nan} for key, val in pd.read_excel(xls, "LTR_US_CCW", index_col=0).to_dict()["LTR_US_CCW"].items()}, "areas": pd.read_excel(xls, "ATR_US_CCW", index_col=0).to_dict()["ATR_US_CCW"]}
pf._results_circ_us_cw = {"lengths": {key: {"length": val, "other": np.nan} for key, val in pd.read_excel(xls, "LTR_US_CW", index_col=0).to_dict()["LTR_US_CW"].items()}, "areas": pd.read_excel(xls, "ATR_US_CW", index_col=0).to_dict()["ATR_US_CW"]}
pf._results_circ_ds_ccw = {"lengths": {key: {"length": val, "other": np.nan} for key, val in pd.read_excel(xls, "LTR_DS_CCW", index_col=0).to_dict()["LTR_DS_CCW"].items()}, "areas": pd.read_excel(xls, "ATR_DS_CCW", index_col=0).to_dict()["ATR_DS_CCW"]}
pf._results_circ_ds_cw = {"lengths": {key: {"length": val, "other": np.nan} for key, val in pd.read_excel(xls, "LTR_DS_CW", index_col=0).to_dict()["LTR_DS_CW"].items()}, "areas": pd.read_excel(xls, "ATR_DS_CW", index_col=0).to_dict()["ATR_DS_CW"]}
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
    OD = 32,
    WT = 0.312,
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
    interaction_weld = False,
    interaction_corrosion = False,
    dent_depth_percent = 2.594,
    ili_pressure = 388,
    restraint_condition = "Restrained",
    ml_depth_percent = 0,
    ml_location = "OD",
    orientation = "Orientation",
    vendor_comments = "FALSE"
)
curve_selection = {"Category": "BS", "Curve": "D", "SD": 0}
cycles = np.array([[3,55,1,3],[4,5,3,4],[5,15,4,5],[15,55,5,15]])
md49_results = mdp.process(dd, pf, (462, 0, 0, cycles, cycles_df.values), curve_selection, False)
print(md49_results)
# Save the results to a txt file
with open("MD53 Validation Results.txt", "w") as f:
    f.write(str(md49_results))
