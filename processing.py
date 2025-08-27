import pandas as pd
import numpy as np
import math
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
import os
import sys
import time
import shutil
import matplotlib.pyplot as plt

def collect_raw_data_v1(rd_path):
    # Vendor 1
    # Works with: Baker Hughes, Enduro, Onestream, Quest, and Rosen

    # Data is all in Column A with delimiter ';' with 12 header rows that need to
    # be removed before accessing the actual data
    rd_axial_row = 12  # Row 13 = Index 12
    rd_drop_tail = 2
    rd = pd.read_csv(rd_path, header=None)
    # Drop the first set of rows that are not being used
    rd.drop(rd.head(rd_axial_row).index, inplace=True)
    # Drop the last two rows that are not being used
    rd.drop(rd.tail(rd_drop_tail).index, inplace=True)
    rd = rd[0].str.split(';', expand=True)
    rd = rd.apply(lambda x: x.str.strip())
    # Drop the last column since it is empty
    rd.drop(rd.columns[-1], axis=1, inplace=True)
    # Relative axial positioning values
    rd_axial = rd.loc[rd_axial_row].to_numpy()
    # Delete the first two values which are 'Offset' and '(ft)'
    rd_axial = np.delete(rd_axial, [0,1])
    rd_axial = rd_axial.astype(float)
    # Convert the axial values to inches
    rd_axial = rd_axial*12
    # Drop the two top rows: Offset and Radius
    rd.drop(rd.head(2).index, inplace=True)
    # Circumferential positioning in [degrees]
    rd_circ = rd[0].to_numpy()
    # Convert from clock to degrees
    rd_circ = [x.split(':') for x in rd_circ]
    rd_circ = [round((float(x[0]) + float(x[1])/60)*360/12,1) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the two first columns: Circumferential in o'Clock and in Length inches
    rd.drop(rd.columns[[0,1]], axis=1, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T

    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v2(rd_path):
    # Vendor 2
    # Works with: 

    rd = pd.read_csv(rd_path, header=None)
    # Axial values are in row direction
    rd_axial = rd[0].to_numpy().astype(float)
    rd_axial = np.delete(rd_axial, 0)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning as caliper number in column direction
    rd_circ = rd.loc[0].to_numpy().astype(float)
    rd_circ = np.delete(rd_circ, 0)
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first row and column
    rd.drop(rd.head(1).index, inplace=True)
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [circ x axial]
    # Make the data negative since it is reading in the opposite direction
    rd_radius = -rd.to_numpy().astype(float)
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v3(rd_path, IR):
    # Vendor 3
    # Works with: TDW

    rd = pd.read_csv(rd_path, header=None)
    # Drop any columns with 'NaN' trailing at the end
    rd.dropna(axis=1, how='all', inplace=True)
    # Drop any columns with ' ' trailing at the end
    if rd.iloc[0][rd.columns[-1]] == ' ':
        rd.drop(columns=rd.columns[-1], axis=1, inplace=True)
    # First row gives the original orientation of each sensor, starting from the second column
    rd_circ = rd.iloc[0][1:].to_numpy().astype(float)
    # Since the orientation values are not incremental, will need to roll the data to have the smallest angle starting
    roll_amount = len(rd_circ) - np.argmin(rd_circ)
    rd_circ = np.roll(rd_circ, roll_amount)
    # Drop the first three rows
    rd.drop(rd.head(3).index, inplace=True)
    # Axial values are in row direction, starting from row 4 (delete first 3 rows)
    rd_axial = rd[0].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    rd_radius = rd.to_numpy().astype(float)
    # Also need to roll the ROWS in rd_radius since the circumferential orientation was rolled
    rd_radius = np.roll(rd_radius, roll_amount, axis=0)
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    # rd_radius = rd_radius - IR
    rd_radius = rd_radius
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v4(rd_path, IR):
    # Vendor 4
    # Works with: Entegra

    rd = pd.read_csv(rd_path, header=None, dtype=float)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in column direction, starting from column B (delete first column)
    rd_axial = rd.loc[0][1:].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning as caliper number in column 
    # Start from 2 since first row is Distance
    rd_circ = rd[0][1:].to_numpy().astype(float)
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first column and row
    rd.drop(rd.columns[0], axis=1, inplace=True)
    rd.drop(rd.head(1).index, axis=0, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    # rd_radius = rd_radius - IR
    # rd_radius = rd_radius
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v5(rd_path, IR):
    # Vendor 5 (created on 09/19/2022)
    # Works with: TDW (similar to original TDW, minus rows 2 and 3)

    rd = pd.read_csv(rd_path, header=None)
    # Drop any columns with 'NaN' trailing at the end
    rd.dropna(axis=1, how='all', inplace=True)
    # Drop any columns with ' ' trailing at the end
    if rd.iloc[0][rd.columns[-1]] == ' ':
        rd.drop(columns=rd.columns[-1], axis=1, inplace=True)
    # First row gives the original orientation of each sensor, starting from the second column
    rd_circ = rd.iloc[0][1:].to_numpy().astype(float)
    # Since the orientation values are not incremental, will need to roll the data to have the smallest angle starting
    roll_amount = len(rd_circ) - np.argmin(rd_circ)
    rd_circ = np.roll(rd_circ, roll_amount)
    # Drop the first row
    rd.drop(rd.head(1).index, inplace=True)
    # Axial values are in row direction, starting from row 2 (delete first 1 row)
    rd_axial = rd[0].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    rd_radius = rd.to_numpy().astype(float)
    # Also need to roll the ROWS in rd_radius since the circumferential orientation was rolled
    rd_radius = np.roll(rd_radius, roll_amount, axis=0)
    # The radial data needs to be the difference form the nominal radius
    # Anything negative means IN and positive means OUT
    rd_radius = rd_radius
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v6(rd_path):
    # Vendor 6 (created on 10/21/2022)
    # Works with: PBF

    # Data has Orientation (oclock) in Horizontal direction starting from B2.
    # Axial information is in Vertical direction, starting from A3
    # Radial values are the IR values
    # Collect raw data and delete the first row
    rd = pd.read_csv(rd_path, header=None, skiprows=1)
    # New first row gives the original orientation of each sensor in oclock, starting from the second column (B)
    rd_circ = rd.iloc[0][1:].to_numpy()
    # Convert from clock to degrees. There are oclock using 12 instead of 0, therefore need to adjust
    rd_circ = [x.split(':') for x in rd_circ]
    rd_circ = [(float(x[0]) * 60 + float(x[1]))/2 for x in rd_circ]
    for i, val in enumerate(rd_circ):
        if val > 360:
            rd_circ[i] = val - 360
    rd_circ = np.array(rd_circ)
    # Since the orientation values are not incremental, will need to roll the data to have the smallest angle starting
    roll_amount = len(rd_circ) - np.argmin(rd_circ)
    rd_circ = np.roll(rd_circ, roll_amount)
    # Drop the first row
    rd.drop(rd.head(1).index, inplace=True)
    # Axial values are in the column direction, starting from A3
    rd_axial = rd[0].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    rd_radius = rd.to_numpy().astype(float)
    # Also need to roll the COLUMNS (axis=1) in rd_radius since the circumferential orientation was rolled
    rd_radius = np.roll(rd_radius, roll_amount, axis=1)
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v7(rd_path):
    # Vendor 7 (created on 04/11/2024)
    # Works with: Southern Company

    # Axial position (m) is in vertical direction, starting at A2
    # There is no Circumferential position or caliper number
    # Internal radial values (mm) start at C2
    rd = pd.read_csv(rd_path, header=None, dtype=float, skiprows=1)
    # Drop column B (index=1) since it is not being used
    rd.drop(rd.columns[1], axis=1, inplace=True)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in COLUMN direction
    rd_axial = rd.loc[0:][0].to_numpy().astype(float)
    # Convert (m) values to relative inches (in)
    rd_axial = [39.3701 * (x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning based on number of caliper data columns
    # Ignore the Axial values column
    rd_circ = np.arange(0, len(rd.loc[0][1:]))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first column
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Collect radial values
    # Important: Data structure needs to be [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float)
    # Convert from (mm) to (in)
    rd_radius = rd_radius * 0.0393701
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v8(rd_path):
    # Vendor 8
    # Works with: Campos

    rd = pd.read_excel(rd_path, header=None)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in column direction, starting from column B (delete first column)
    rd_axial = rd.loc[0][1:].to_numpy().astype(float)
    # Convert ft values to relative inches
    rd_axial = [12*(x - rd_axial[0]) for x in rd_axial]
    rd_axial = np.array(rd_axial)
    # Circumferential positioning as caliper number in column 
    # Start from 2 since first row is Distance
    # rd_circ = rd[0][1:].to_numpy().astype(float)
    rd_circ = rd[0][1:]
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = [(x*360/len(rd_circ)) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the first column and row
    rd.drop(rd.columns[0], axis=1, inplace=True)
    rd.drop(rd.head(1).index, axis=0, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T
    # Convert from mm to inches
    rd_radius = rd_radius / 25.4
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v9(rd_path):
    # Vendor 9
    # Works with: Creaform Scan

    rd = pd.read_csv(rd_path, header=None)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in column direction, starting from column B (delete first column)
    rd_axial = rd.loc[0][1:].to_numpy().astype(float)
    # Circumferential values (deg) are in row direction, starting from row 2 (delete first row)
    rd_circ = rd.loc[1:][0].to_numpy().astype(float)
    # Drop the first column and row
    rd.drop(rd.columns[0], axis=1, inplace=True)
    rd.drop(rd.head(1).index, axis=0, inplace=True)
    # Collect radial values and make sure to transpose so that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float).T
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v10(rd_path):
    # Vendor 10
    # Works with: Rosen

    # Collect raw data
    rd = pd.read_csv(rd_path, header=None, dtype=float, skiprows=1)
    # Drop any columns and rows with 'NaN' trailing at the end
    rd = rd.dropna(axis=1, how='all')
    rd = rd.dropna(axis=0, how='all')
    # Axial values are in column direction (delete second column)
    rd_axial = rd.loc[:][0].to_numpy().astype(float)
    # Convert [m] values to relative [in]
    rd_axial = np.array([39.3701 *(x - rd_axial[0]) for x in rd_axial])
    # Drop columns A and B
    rd.drop(rd.columns[0], axis=1, inplace=True)
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Circumferential positioning as caliper number in column 
    # Start from 2 since first row is Distance
    rd_circ = rd.loc[0].to_numpy().astype(float)
    # Since circumferential positioning may not be in the numerical order
    rd_circ = np.arange(0, len(rd_circ))
    # Convert from number to degrees
    rd_circ = np.array([(x*360/len(rd_circ)) for x in rd_circ])
    # Collect radial values and make sure that [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float)
    # Convert from [mm] to [in]
    rd_radius = rd_radius / 25.4
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v11(rd_path):
    # Vendor 11
    # Works with: KS12

    # Collect raw data
    rd = pd.read_csv(rd_path, header=0, dtype=float)
    # Column index 82 contains continuous measurements of internal radius (IR) in mm
    IR = rd.iloc[:, 82] / 25.4  # Convert from mm to inches
    # Drop columns after column index 80
    rd.drop(rd.columns[81:], axis=1, inplace=True)
    # Make the first column the axial values (convert from meters to relative inches), and drop the first column
    rd_axial = rd[rd.columns[0]].to_numpy().astype(float) * 3.28084 * 12
    rd_axial = rd_axial - rd_axial[0]
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Extract the Caliper Number from the column headers, format is "Deflection XX (mm)"
    rd_caliper = [int(col.split(' ')[1]) for col in rd.columns]
    # Convert from caliper number to degrees, but in numerical order
    rd_circ = np.array([(x*360/len(rd_caliper)) for x in range(len(rd_caliper))])
    # Collect the radial values, convert from mm to inches, and ensure that the data structure is [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float) / 25.4  # Convert from mm to inches
    # Adjust each row of rd_radius by adding the IR value to the negative of rd_radius (values are negative direction)
    for i in range(rd_radius.shape[0]):
        rd_radius[i, :] = -rd_radius[i, :] + IR[i]
    return rd_axial, rd_circ, rd_radius

def collect_raw_data_v12(rd_path):
    # Vendor 12
    # Works with: KS1314

    # Collect raw data
    with open(rd_path, 'rb') as f:
        rd = pd.read_excel(f, header=0, dtype=float, engine='calamine')
    # Column index 0 contains the axial values (convert from mm to relative inches), and drop the first column
    rd_axial = rd[rd.columns[0]].to_numpy().astype(float) * 0.0393701  # Convert from mm to inches
    rd_axial = rd_axial - rd_axial[0]
    rd.drop(rd.columns[0], axis=1, inplace=True)
    # Extract the circumferential degrees from the column headers
    rd_circ = rd.columns.to_numpy().astype(float)
    # Confirm that values in rd_circ are in incremental order, otherwise, redetermine the values based on the number of columns
    if not np.all(np.diff(rd_circ) > 0):
        rd_circ = np.linspace(0, 360, num=rd.shape[1], endpoint=False)
    # Collect the radial values, convert from mm to inches, and ensure that the data structure is [r x c] = [axial x circ]
    rd_radius = rd.to_numpy().astype(float) / 25.4  # Convert from mm to inches
    return rd_axial, rd_circ, rd_radius


class Process:
    def __init__(self, rd_path, ILI_format, OD, WT, SMYS, filename):
        """
        Import data in the desired ILI format. Below are a list of recognized ILI formats.

        Parameters
        ----------
        rd_path : string
            the location of the feature of interest
        ILI_format : string
            the desired ILI format that corresponds to the feature of interest
        OD : float
            the outside diameter measurement, in
        Outputs
        ----------
        o_axial & f_axial : array of floats
            1-D array containing the axial displacement, in
        o_circ & f_circ : array of floats
            1-D array containing the circumferential displacement, deg
        o_radius & f_radius : array of floats
            2-D array containing the radial values with shape (axial x circ), in
        """
        # self.name = rd_path.split('/')[-1].split('.')[0]    # Get the filename
        self.name = filename.split('.')[0]
        self.path = rd_path
        self.ILI_format = str(ILI_format)
        self.OD = OD
        self.WT = WT
        self.SMYS = SMYS
        self.input_file = False

        ILI_format_list = ['Baker Hughes', 'Enduro', 'Entegra', 'Onestream', 'Quest',
                           'Rosen', 'TDW', 'TDW (v2)', 'PBF', 'Campos', 'Southern', 'Creaform']

        # Load the raw data information
        if ILI_format.lower() == 'baker hughes':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'enduro':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'entegra':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v4(rd_path, OD/2)
        elif ILI_format.lower() == 'onestream':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'quest':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'rosen':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
        elif ILI_format.lower() == 'tdw':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v3(rd_path, OD/2)
        elif ILI_format.lower() == 'tdw2':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v5(rd_path, OD/2)
        elif ILI_format.lower() == 'pbf':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v6(rd_path)
        elif ILI_format.lower() == 'campos':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v8(rd_path)
        elif ILI_format.lower() == 'southern':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v7(rd_path)
        elif ILI_format.lower() == 'creaform':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v9(rd_path)
        elif ILI_format.lower() == 'rosen2':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v10(rd_path)
        elif ILI_format.lower() == 'ks12':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v11(rd_path)
        elif ILI_format.lower() == 'ks1314':
            rd_axial, rd_circ, rd_radius = collect_raw_data_v12(rd_path)
        else:
            raise Exception('ILI format %s was not found. Use one of the following: %s' % (ILI_format, ', '.join(ILI_format_list)))
        
        # Keep the original
        self.o_axial = rd_axial
        self.o_circ = rd_circ
        self.o_radius = rd_radius
        # Make a final copy for manipulation
        self.f_axial = rd_axial
        self.f_circ = rd_circ
        self.f_radius = rd_radius

    def smooth_data(self, circ_int=0.5, axial_int=0.5, circ_window=5, circ_smooth=0.001, axial_window=9, axial_smooth=0.00005):
        """
        ASME B31.8-2020 Nonmandatory Appendix R Estimating Strain in Dents recommends 
        the use of suitable data smoothing techniques in order to minimize the effect 
        of random error inherent with all geometric measurement techniques.
        
        This function applies the Savitzky-Golay filter on the data, then generates 
        spline curves that are evaluated at desired intervals.

        Parameters
        ----------
        OD : float
            pipeline nominal outside diameter, in
        rd_axial : array of floats
            1-D array containing the axial displacement, in
        rd_circ : array of floats
            1-D array containing the circumferential displacement, deg
        rd_radius : array of floats
            2-D array containing the radial values with shape (axial x circ), in
        circ_int : float
            the desired circumferential interval length for the output data, in. Default = 0.5
        axial_int : float
            the desired axial interval length for the output data, in. Default = 0.5
        circ_window : int
            the smoothing window (number of points to consider) for the circumferential 
            smoothing filter. Note: this must be an odd number. Default = 5
        circ_smooth : float
            the circumferential smoothing parameter for splines. Default = 0.001
        axial_window : int
            the smoothing window (number of points to consider) for the axial 
            smoothing filter. Note: this must be an odd number. Default = 9
        axial_smooth : float
            the axial smoothing parameter for splines. Default 0.00005
        
        Returns
        -------
        sd_axial : array of floats
            the smoothed axial displacement values in the fixed intervals, in.
        sd_circ : array of floats
            the smoothed circumferential displacement values in the fixed intervals, in.
        sd_radius : array of floats
            the smoothed radial values in the fixed intervals, in.
        """
        
        # Always input the original data to ensure that data is not smoothed twice
        rd_axial = self.o_axial
        rd_circ_deg = self.o_circ
        rd_radius = self.o_radius
        
        filter_polyorder = 3
        filter_mode = 'wrap'
        spline_deg = 3
        
        OR = self.OD/2
        
        # Convert the circumferential orientation from degrees to radians
        rd_circ = np.deg2rad(rd_circ_deg)
        
        # Smoothed file output interval to 0.50" by 0.50"
        int_len_circ     = circ_int      # Circumferential interval length, in
        int_len_axial    = axial_int     # Axial interval length, in
        
        int_count_circ   = math.ceil((2*math.pi*OR/int_len_circ)/4)*4                       # Find the circumferential interval closest to int_len_circ on a multiple of four
        int_count_axial  = int(max(rd_axial) / int_len_axial)                               # Find the axial interval closest to int_len_axial
        
        int_points_circ  = np.linspace(0, 2*math.pi, int_count_circ, False)                 # Create circumferential interval for one pipe circumference in radians
        int_points_axial = np.linspace(0, round(max(rd_axial), 0), int_count_axial, False)  # Create equally spaced axial points for smoothing
        
        sd_radius_circ1  = np.zeros(rd_radius.shape)                                        # First pass of smoothing will have the same number of data points as the raw data
        sd_radius_axial1 = np.zeros(rd_radius.shape)                                        # First pass of smoothing will have the same number of data points as the raw data
        
        sd_radius_circ2  = np.zeros((len(rd_axial), len(int_points_circ)))                  # Second pass of smoothing will have the desired interval number of data points
        sd_radius_axial2 = np.zeros((len(int_points_axial), len(int_points_circ)))          # Second pass of smoothing will have the desired interval number of data points
        
        # Step 1: Circumferential profiles smoothing and spline functions
        for axial_index, circ_profile in enumerate(rd_radius[:,0]):
            circ_profile    = rd_radius[axial_index, :]
            circ_filter     = savgol_filter(x=circ_profile, window_length=circ_window, polyorder=filter_polyorder, mode=filter_mode)
            circ_spline     = splrep(x=rd_circ, y=circ_filter, k=spline_deg, s=circ_smooth, per=1) # Data is considered periodic since it wraps around, therefore per=1
            sd_radius_circ1[axial_index, :] = splev(x=rd_circ, tck=circ_spline)
            
        # Step 2: Axial profiles smoothing and spline functions
        for circ_index, axial_profile in enumerate(rd_radius[0,:]):
            axial_profile   = rd_radius[:, circ_index]
            axial_filter    = savgol_filter(x=axial_profile, window_length=axial_window, polyorder=filter_polyorder)
            axial_spline    = splrep(x=rd_axial, y=axial_filter, k=spline_deg, s=axial_smooth)
            sd_radius_axial1[:, circ_index] = splev(x=rd_axial, tck=axial_spline)
            
        # Step 3: Create weighted average profiles from axial and circumferential profiles
        circ_err    = abs(sd_radius_circ1 - rd_radius)
        axial_err   = abs(sd_radius_axial1 - rd_radius)
        sd_radius_avg   = (circ_err * sd_radius_circ1 + axial_err * sd_radius_axial1)/(circ_err + axial_err)
        
        # Step 4: Final profiles with the desired intervals, starting with axial direction
        for axial_index, circ_profile in enumerate(sd_radius_avg[:,0]):
            circ_profile    = sd_radius_avg[axial_index, :]
            # circ_filter     = savgol_filter(x=circ_profile, window_length=circ_window, polyorder=filter_polyorder, mode=filter_mode) # Added this line for testing 04/18/2024
            circ_spline     = splrep(x=rd_circ, y=circ_profile, k=spline_deg, s=circ_smooth, per=1)
            sd_radius_circ2[axial_index, :] = splev(x=int_points_circ, tck=circ_spline)
            
        for circ_index, axial_profile in enumerate(sd_radius_circ2[0,:]):
            axial_profile = sd_radius_circ2[:, circ_index]
            axial_filter = savgol_filter(x=axial_profile, window_length=axial_window, polyorder=filter_polyorder)
            axial_spline = splrep(x=rd_axial, y=axial_filter, k=spline_deg, s=axial_smooth)
            sd_radius_axial2[:, circ_index] = splev(x=int_points_axial, tck=axial_spline)
        
        sd_axial  = int_points_axial
        sd_circ   = np.rad2deg(int_points_circ)
        sd_radius = sd_radius_axial2
        
        # Save parameters
        self.circ_int = circ_int
        self.axial_int = axial_int
        self.circ_window = circ_window
        self.circ_smooth = circ_smooth
        self.axial_window = axial_window
        self.axial_smooth = axial_smooth

        self.f_axial = sd_axial
        self.f_circ = sd_circ
        self.f_radius = sd_radius

    def create_input_file(self, results_path='results/', templates_path='templates/'):
        dent_ID     = self.name
        OD          = self.OD
        sd_axial    = self.f_axial
        sd_circ     = self.f_circ
        sd_radius   = self.f_radius
        inp_wt      = self.WT
        num_cal     = sd_circ.size
        num_nodes   = sd_radius.size
        def_angl    = 60
        bar_stress  = self.SMYS # This is based on the SMYS of the pipe

        # Create an output folder for the dent analysis results.
        results_path = results_path + str(dent_ID)
        try:
            # Directory does not exist, make a new folder
            os.makedirs(results_path, exist_ok=False)
        except:
            # Directory already exists
            sys.exit(f'Directory {results_path} already exists. Please enter a new folder location or delete the exist Results folder.')
        self.results_path = results_path + '/'

        # lim_cc and lim_ax is the amount of nodes to display applied to both sides in the circumferential and axial directions, respectively
        # For example, using lim_cc = 20 and lim_ax 40 will result in a field of points of (circ x axial) = (40 x 80)
        # lim_cc needs to be half of the circumference
        # lim_cc      = 20
        # circ_interval = []
        # for i in range(len(sd_circ)-1):
        #     circ_interval.append(sd_circ[i+1] - sd_circ[i])
        # circ_interval_avg = np.deg2rad(np.mean(circ_interval))
        # lim_cc = math.ceil(1/2*2*math.pi*(OD/2)/circ_interval_avg/2)
        lim_cc = int(sd_circ.shape[0]/4)
        
        # lim_ax needs to be a span of 2*OD of the axial
        # lim_ax      = 40
        axial_interval = []
        for i in range(len(sd_axial)-1):
            axial_interval.append(sd_axial[i+1] - sd_axial[i])
        axial_interval_avg = np.mean(axial_interval)
        lim_ax = int(math.ceil(2*OD/axial_interval_avg/2))
        
        # Create the *Node array
        z_len = sd_axial.size
        theta_len = sd_circ.size
        inp_num_nodes = sd_radius.size
        
        inp_node = []
        inp_node_i = 0
        for iz in range(0,z_len):
            for it in range(0, theta_len):
                inp_node.append(str(inp_node_i + 1) + ", " + str(round(sd_radius[iz,it],3)) + ", " + str(round(sd_circ[it],3)) + ", " + str(round(sd_axial[iz],3)))
                inp_node_i += 1
        
        # Create the *Element and *Elgen arrays
        el1 = 0
        el4 = 0
        j = 0
        inp_element = []
        inp_elgen = []
        
        theta_len = sd_circ.size
        while el4 < inp_node_i:
            # A
            j += 1
            el1 += 1
            el2 = el1 + 1
            el3 = el2 + theta_len
            el4 = el3 - 1
            inp_element.append(str(el1)+", "+str(el1)+", "+str(el2)+", "+str(el3)+", "+str(el4))
            inp_elgen.append(str(el1)+", "+str(theta_len - 1)+", 1, 1")
            # B
            j += 1
            el2 = el1
            el3 = el2 + theta_len
            el4 = el3 + theta_len - 1
            el1 = el1 + theta_len - 1
            inp_element.append(str(el1)+", "+str(el1)+", "+str(el2)+", "+str(el3)+", "+str(el4))
            inp_elgen.append(str(el1)+", 1, 1, 1")
        
        # Create the boundary condition nodes, *BCNodes
        # The first set of nodes at the start Z position, and the second at the last Z position
        inp_bcnode = list(range(1, theta_len + 1)) + list(range(inp_node_i - theta_len, inp_node_i + 1))
        
        # Create the indenter shape using R1 (Circumferential Radius) and R2 (Axial Radius) from the deepest part of the dent
        sd_circ_rad = np.deg2rad(sd_circ)
        # [Index, Value] in Circumferential Direction
        min_ind = np.unravel_index(np.argmin(sd_radius), sd_radius.shape)

        # Select the circumferential profile including the deepest part of the dent
        circ_profile = sd_radius[min_ind[0], :]
        # First derivative
        d_circ = np.gradient(circ_profile, sd_circ_rad)
        # Second derivative
        dd_circ = np.gradient(d_circ, sd_circ_rad)
        # Radius of curvature in polar coordinates, select only the one corresponding to the deepest dent
        R1 = (circ_profile**2 + d_circ**2)**(3/2)/abs(circ_profile**2 + 2*d_circ**2 - circ_profile*dd_circ)
        R1 = R1[min_ind[1]]

        # Select the axial profile including the deepest part of the dent
        axial_profile = sd_radius[:, min_ind[1]]
        # First derivative
        d_axial = np.gradient(axial_profile, sd_axial)
        # Second derivative
        dd_axial = np.gradient(d_axial, sd_axial)
        # Radius of curvature, select only the one corresponding to the deepest dent
        R2 = (1 + d_axial**2)**(3/2)/np.float64(abs(dd_axial))
        R2[R2 == np.inf] = 1000000
        R2 = R2[min_ind[0]]

        # Equation for the ellipsoid
        d = 1.35
        A = np.sqrt(R2*d)
        B = np.sqrt(R1*d)
        xx = np.linspace(-A,A,50)
        yy = np.linspace(-B,B,50)
        fun = d**2 * (1 - xx**2/A**2 - yy**2/B**2)
        fun[fun < 0] = 0
        Z = -np.sqrt(fun)
        Z = Z + abs(np.min(Z))

        # Generate a mesh by taking axial cross-sections of the ellipsoid
        
        # Loop through the inp_file and search for the following keywords
        # - #Nodes#
        # - #Elements#
        # - #BCNodes#
        # - #Elgen#
        # - #All_Elements#
        # - #Wall_Thickness#
        
        # Create a copy of the Input Deck Template text file
        inp_file_template_str = templates_path + 'Input Deck Template.inp'
        self.inp_file_name = "Feature_" + str(dent_ID)
        self.inp_file_path = self.results_path + 'FEA Results'
        # Create a folder for the Abaqus files
        os.mkdir(self.inp_file_path)
        self.inp_file_path = self.inp_file_path + '/Abaqus/'
        inp_file_deck = self.inp_file_path + self.inp_file_name + '.inp'
        # Load the Input Deck Template text file
        inp_file_template = open(inp_file_template_str, 'r')
        # Create a leaf directory and all intermediate ones
        # Each input deck will have its own folder to store all Abaqus files
        os.makedirs(os.path.dirname(inp_file_deck), exist_ok=True)
        inp_file = open(inp_file_deck, 'w')
        
        # Keep track of the placeholder strings to later remove them
        inp_line_list = ["#Nodes#\n","#Elements#\n","#Elgen#\n","#BCNodes#\n","#All_Elements#\n","#All_Elements#\n","#Wall_Thickness#, 5\n"]
        inp_line_index = []
        inp_file_contents = inp_file_template.readlines()
        for f_index, line in enumerate(inp_file_contents):
            # Search for #Nodes#
            if line == inp_line_list[0]:
                # Print all of the inp_node values to the Input Deck File
                for n_index, n_value in enumerate(inp_node):
                    inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
            # Search for #Elements#
            if line == inp_line_list[1]:
                # Print all of the inp_node values to the Input Deck File
                for n_index, n_value in enumerate(inp_element):
                    inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
            # Search for #Elgen#
            if line == inp_line_list[2]:
                # Print all of the inp_node values to the Input Deck File
                for n_index, n_value in enumerate(inp_elgen):
                    inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
            # Search for #BCNodes#
            if line == inp_line_list[3]:
                # Print all of the inp_node values to the Input Deck File
                for n_index, n_value in enumerate(inp_bcnode):
                    inp_file_contents.insert(f_index + n_index + 1, str(n_value) + "\n")
            # Search for #All_Elements#
            if line == inp_line_list[4]:
                # Print the value for #All_Elements#
                inp_line_index.append(f_index)
                inp_file_contents.insert(f_index + 1,"1, "+str(inp_num_nodes - theta_len + 1)+", 1\n")
            # Search for #Wall_Thickness#
            if "#Wall_Thickness#" in line:
                # Print the value for #Wall_Thickness#
                inp_line_index.append(f_index)
                inp_file_contents.insert(f_index + 1,str(inp_wt)+", 5 \n")
            if "#Pressure#" in line:
                # Print the value for #Pressure#
                inp_line_index.append(f_index)
                bar_press = (2*self.WT*bar_stress)/OD
                inp_file_contents[f_index] = inp_file_contents[f_index].replace("#Pressure#", str(round(bar_press,4)))
        # Remove the placeholder strings
        for i in inp_line_list:
            inp_file_contents.remove(i)
        
        inp_file.writelines(inp_file_contents)
        inp_file.close()
        
        # I need to print a file with the essential information needed in the Post-Processing section
        # Part C: Isolated Elements View
        # cc_lim    - Circumferential Limit
        # ax_lim    - Axial Limit
        # num_cal   - Number of Calipers
        # num_nodes - Total Number of Nodes
        # def_angl  - Angle for Isometric View
        
        # Create a node_info.txt file to export theses values
        info_file_name = "node_info"
        info_file_deck = self.inp_file_path + info_file_name + ".txt"
        info_file = open(info_file_deck, "w")
        # Data to write in
        info_file_contents = []
        # # Feature information
        # info_file_contents.append('======== NODE INFORMATION ========' + '\n')
        # info_file_contents.append('Feature ID = ' + str(dent_ID) + '\n\n')
        # info_file_contents.append('The Circumferential Limit (lim_cc) and Axial Limit (lim_ax) specify the window of nodes to display in the Internal Review.\n')
        # info_file_contents.append('For example, using lim_cc = 20 and lim_ax 40 will result in a field of points of (circ x axial) = (40 x 80)\n')
        # info_file_contents.append('For best results, it is recommended to use a field of view containing half of the circumference and 2*OD of the axial.\n')
        # info_file_contents.append('============= VALUES =============' + '\n')
        # lim_cc - Circumferential Limit
        info_file_contents.append('lim_cc     = ' + str(lim_cc) + "\n")
        # lim_ax - Axial Limit
        info_file_contents.append('lim_ax     = ' + str(lim_ax) + "\n")
        # num_cal - Number of Calipers
        info_file_contents.append('num_cal    = ' + str(num_cal) + "\n")
        # num_nodes - Total Number of Nodes
        info_file_contents.append('num_nodes  = ' + str(num_nodes) + "\n")
        # def_angl - Angle for Isometric View
        info_file_contents.append('def_angl   = ' + str(def_angl) + "\n")
        # bar_stress - Barlow's equation for Hoop Stress to calculate SCF
        info_file_contents.append('bar_stress = ' + str(bar_stress) + "\n")
        info_file.writelines(info_file_contents)
        info_file.close()

        self.input_file = True

    def submit_input_file(self, templates_path='templates/'):
        if self.input_file == False:
            return print('Input file has not been created. Please create the input file first.')

        inp_file_name = self.inp_file_name
        inp_file_path = self.inp_file_path

        time_start = time.time()
        time_ref = '%03d | '
        time_limit = 60*20 # 20 minutes

        # In order to maintain the same Command Prompt environment, need to do both the
        # cd directory change and the Abaqus command in one os.system wrapper.
        command_str = "abaqus job=" + inp_file_name + " cpus=2"
        # command_dir = "cd " + os.getcwd() + "/" + inp_file_path
        command_dir = "cd " + inp_file_path
        command = command_dir + " && " + command_str
        # Clear the existing history before running the command
        os.system('cls')
        os.system('C:')
        os.system(command)
        os.system('cls')

        # First check that the file exists, since there may be a delay before it is created
        sta_path = inp_file_path + inp_file_name + ".sta"
        file_check_time = time.time()
        
        while not os.path.exists(sta_path):
            time.sleep(5)
            print((time_ref + 'Waiting for Abaqus to create the .sta file.') % (time.time() - time_start))
            
            # Check that it has not exceeded the time limit to prevent an infite loop
            if (time.time() - file_check_time)>time_limit:
                # End the script
                print((time_ref + '========== ERROR ==========') % (time.time() - time_start))
                sys.exit('Exceeded time limit of %.0f seconds to search for .sta file. Proceess aborted.' % (time_limit))
            
        print((time_ref + 'Abaqus has created the .sta file. Begin monitoring this file until Abaqus concludes.') % (time.time() - time_start))
        
        # Restart the time limit for monitoring the .sta file
        file_check_time = time.time()
        
        sta_file = open(sta_path, "r")
        sta_contents = sta_file.readlines()
        
        # Loop until " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n" shows up at the end
        str_success = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n"
        while sta_contents[-1] != str_success:
            # Wait for 30 seconds before opening the file again and checking
            time.sleep(10)
            # Reload the .sta file with its contents
            print((time_ref + 'Monitoring the .sta file.') % (time.time() - time_start))
            sta_file = open(sta_path, "r")
            sta_contents = sta_file.readlines()
            
            # Check that it has not exceeded the time limit to prevent an infite loop
            if (time.time() - file_check_time)>time_limit:
                # End the script
                print((time_ref + '========== ERROR ==========') % (time.time() - time_start))
                sys.exit('Exceeded time limit of %.0f seconds to search for .sta file. Proceess aborted.' % (time_limit))
        
        sta_file.close()
        print((time_ref + '===== SCF CALCULATION ======') % (time.time() - time_start))
        
        # Copy the abaqusMacros.py template file to the input deck folder
        script_path = templates_path
        script_name = 'abaqusMacros.py'
        script_file = script_path + script_name
        # Use the same destination as the Input File
        shutil.copy(script_file, inp_file_path)
        # Run the abaqusMacros.py script to do the following:
        # - Print 11 images from the .odb file for analysis
        # - Print the MaxPrincipal value for future SCF calculations
        command_str = "abaqus viewer noGUI=" + script_name
        command_dir = "cd " + inp_file_path
        command = command_dir + " && " + command_str

        os.system('cls')
        os.system('C:')
        os.system(command)
        os.system('cls')
        
        # Wait for 30 seconds for all images to be created
        time.sleep(15)

    def calculate_strain(self, d, L):
        """
        ASME B31.8-2020 Nonmandatory Appendix R Estimating Strain in Dents calculates
        the bending strain in the circumferential direction, e1, the bending strain in
        the longitudinal direction, e2, and the extensional strain in the longitudinal
        direction, e3. 
        
        This function calculates strains e1, e2, and e3 along with the strain for the
        inside and outside pipe surfaces.

        Returns
        -------
        df_eo : DataFrame
            DataFrame containing all of the strain for the outside pipe surface
        df_ei : DataFrame
            DataFrame containing all of the strain for the inside pipe surface
        df_e1 : DataFrame
            DataFrame containing the bending strain in the circumferential direction
        df_e2 : DataFrame
            DataFrame containing the bending strain in the longitudinal direction
        e3 : float
            float value of the extensional strain in the longitudinal direction
        df_R1 : DataFrame
            DataFrame containing the Radius of Curvature in the circumferential plane
        df_R2 : DataFrame
            DataFrame containing the Radius of Curvature in the longitudinal plane
        """
        self.d = d
        self.L = L
        circ_r = np.deg2rad(self.f_circ)

        R0 = self.OD/2

        # Strain calculations
        sd_e1 = np.zeros(self.f_radius.shape)
        sd_e2 = np.zeros(self.f_radius.shape)

        e3 = (1/2)*(self.d/self.L)**2

        # Radius of curvatures
        sd_R1 = np.zeros(self.f_radius.shape)
        sd_R2 = np.zeros(self.f_radius.shape)
        
        # Calculate the bending strain in the circumferential direction, e1
        for axial_index, circ_profile in enumerate(self.f_radius[:,0]):
            circ_profile = self.f_radius[axial_index, :]
            # First derivative
            d_circ = np.gradient(circ_profile, circ_r)
            # Second derivative
            dd_circ = np.gradient(d_circ, circ_r)
            # Radius of curvature in polar coordinates
            R1 = (circ_profile**2 + d_circ**2)**(3/2)/abs(circ_profile**2 + 2*d_circ**2 - circ_profile*dd_circ)
            # Calculate e1 and save it for this circumferential profile
            sd_e1[axial_index, :] = (self.WT/2)*(1/R0 - 1/R1)
            sd_R1[axial_index, :] = R1
            
        # Calculate the bending strain in the longitudinal (axial) direction, e2
        for circ_index, axial_profile in enumerate(self.f_radius[0,:]):
            axial_profile = self.f_radius[:, circ_index]
            # First derivative
            d_axial = np.gradient(axial_profile, self.f_axial)
            # Second derivative
            dd_axial = np.gradient(d_axial, self.f_axial)
            # Radius of curvature. Added np.float64 to help division by zero -> inf
            R2 = (1 + d_axial**2)**(3/2)/np.float64(abs(dd_axial))
            R2[R2 == np.inf] = 1000000
            # Calculate e2 and save it for this axial profile
            sd_e2[:, circ_index] = self.WT/(2*R2)
            sd_R2[:, circ_index] = R2
            
        # Calculate the final strain for the outside pipe, eo, and inside pipe, ei
        self.ei = (2/np.sqrt(3))*np.sqrt(sd_e1**2 + sd_e1*(sd_e2 + e3) + (sd_e2 + e3)**2)
        self.eo = (2/np.sqrt(3))*np.sqrt((-sd_e1)**2 + (-sd_e1)*((-sd_e2) + e3) + ((-sd_e2) + e3)**2)
        self.e1 = sd_e1
        self.e2 = sd_e2
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

    def review_abaqus_results(self):
        file_new_name = self.inp_file_name + '_'
        # Get the FEA Results directory
        fea_path = self.results_path + 'FEA Results/'
        # fea_path = self.inp_file_path
        
        # Step 1: Read the report_MPs.rpt
        with open(fea_path + 'report_MPs.rpt') as f:
            report_MPs = f.readlines()
            f.close()
        # Delete the first 3 rows since they do not contain data
        report_MPs.pop(0)
        report_MPs.pop(0)
        report_MPs.pop(0)
        # Remove the leading and trailing spaces, also the '\n' character
        report_MPs = [s.strip() for s in report_MPs]
        # Split by spaces
        report_MPs = [s.split(" ") for s in report_MPs]
        # Remove the empty list items
        report_MPs = [s for s in report_MPs if s != ['']]
        # Remove the spaces and convert string numbers into floats
        x1 = np.zeros(len(report_MPs))
        yMPs = np.zeros(len(report_MPs))
        for i in range(0,len(report_MPs)):
            report_MPs[i] = [float(s) for s in report_MPs[i] if s.strip()]
            x1[i] = report_MPs[i][0]
            yMPs[i] = report_MPs[i][1]
        
        # Step 2: Read the report_Radius.rpt
        with open(fea_path + 'report_Radius.rpt') as f:
            report_Radius = f.readlines()
            f.close()
        # Delete the first 3 rows since they do not contain data
        report_Radius.pop(0)
        report_Radius.pop(0)
        report_Radius.pop(0)
        # Remove the leading and trailing spaces, also the '\n' character
        report_Radius = [s.strip() for s in report_Radius]
        # Split by spaces
        report_Radius = [s.split(" ") for s in report_Radius]
        # Remove the empty list items
        report_Radius = [s for s in report_Radius if s != ['']]
        # Remove the spaces and convert string numbers into floats
        x2 = np.zeros(len(report_Radius))
        R = np.zeros(len(report_Radius))
        for i in range(0,len(report_Radius)):
            report_Radius[i] = [float(s) for s in report_Radius[i] if s.strip()]
            x2[i] = report_Radius[i][0]
            R[i] = report_Radius[i][1]

        # Step 3: Read the report_All_Data.rpt
        with open(fea_path + 'report_All_Data.rpt') as f:
            report_All_Data = f.readlines()
            f.close()
        # Delete the first 26 rows since they do not contain data
        report_All_Data = report_All_Data[26:-10]
        # Remove the leading and trailing spaces, also the '\n' character
        report_All_Data = [s.strip() for s in report_All_Data]
        # Split by spaces
        report_All_Data = [s.split(" ") for s in report_All_Data]
        # Remove the empty list items
        report_All_Data = [list(filter(None, s)) for s in report_All_Data if s !=['']]
        # Convert string numbers into floats
        report_All_Data = np.array(report_All_Data, dtype='float')

        self.COORD = np.zeros(self.f_radius.shape)
        self.LE_SNEG = np.zeros(self.f_radius.shape)
        self.LE_SPOS = np.zeros(self.f_radius.shape)
        self.S_SNEG = np.zeros(self.f_radius.shape)
        self.S_SPOS = np.zeros(self.f_radius.shape)

        # DRAFT CODE STARTS FOR MATCHING NODE INDEX
        iall = 0
        for iz in range(0,self.f_axial.size):
            for it in range(0, self.f_circ.size):
                self.COORD[iz, it] = report_All_Data[iall,1]
                self.LE_SNEG[iz, it] = report_All_Data[iall, 2]
                self.LE_SPOS[iz, it] = report_All_Data[iall, 3]
                self.S_SNEG[iz, it] = report_All_Data[iall, 4]
                self.S_SPOS[iz, it] = report_All_Data[iall, 5]
                # inp_node.append(str(inp_node_i + 1) + ", " + str(round(sd_radius[iz,it],3)) + ", " + str(round(sd_circ[it],3)) + ", " + str(round(sd_axial[iz],3)))
                # inp_node_i += 1
                iall += 1
        # DRAFT CODE ENDS
        
        
        # # Produce the Node Path Image
        # plt.rcParams['font.size'] = 8
        # plt.rcParams['lines.markersize'] = 0.5
        
        # fig9, ax9 = plt.subplots(figsize=(3.43,2), dpi=200)
        # ax9_1 = ax9.twinx()
        # # First Y Axis
        # ax9.plot(x1, yMPs, c='tab:blue', label='Max. Principal Stress',
        #         marker='o', markerfacecolor='k', markeredgecolor='k', markersize=1)
        # ax9.set_xlabel('Position Z Along Pipe [in]')
        # ax9.set_ylabel('Maximum Principal Stress [psi]')
        # ax9.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        # ax9.tick_params(axis='y', colors='tab:blue')
        # ax9.yaxis.label.set_color('tab:blue')
        # # Secondary Y Axis
        # ax9_1.plot(x2, R, c='tab:orange',label='Axial Radial Profile',
        #         marker='o', markerfacecolor='r', markeredgecolor='r', markersize=1)
        # ax9_1.set_ylabel('Radius [in]')
        # ax9_1.tick_params(axis='y', colors='tab:orange')
        # ax9_1.yaxis.label.set_color('tab:orange')
        # # Save as the OR version
        # fig9.savefig(fea_path + file_new_name + '14_Nodal_Path_OR', bbox_inches='tight')
        
        # # Adjust so that it saves as the full size
        # plt.rcParams['font.size'] = 8
        # plt.rcParams['lines.markersize'] = 0.5
        # fig9.set_size_inches(10,4)
        # fig9.suptitle('Max. Principal Stress OD along Axial Radial Profile')
        # s1,sl1 = ax9.get_legend_handles_labels()
        # s2,sl2 = ax9_1.get_legend_handles_labels()
        # s = s1 + s2
        # sl = sl1 + sl2
        # ax9.legend(s,sl)
        # fig9.savefig(fea_path + file_new_name + '14_Nodal_Path', dpi=200)
        # # fig9.savefig('all_results/' + file_new_name + '_14_Nodal_Path.png')
        
        # # Collect the SCF value from the information file
        # with open(fea_path + "node_info.txt") as f:
        #     node_info = f.readlines()
        #     f.close()
        
        # node_val = [s.strip() for s in node_info]
        # node_val = [s.split("=") for s in node_val if "=" in s]
        # # Collect the Max Principal Stress Value in the OD
        # MPs = [float(s[1]) for s in node_val if "max_val_OD" in s[0]]
        # MPs = float(MPs[0])
        # # Collect the SCF Values (for ID and OD), but only keep the largest
        # SCF_ID = [float(s[1]) for s in node_val if "ID SCF" in s[0]]
        # SCF_ID = float(SCF_ID[0])
        # SCF_OD = [float(s[1]) for s in node_val if "OD SCF" in s[0]]
        # SCF_OD = float(SCF_OD[0])
        # SCF = max(SCF_ID, SCF_OD)
        # # Collect the Unaveraged Max Principal Stress Value
        # uMPs = [float(s[1]) for s in node_val if "Unavg MPs" in s[0]]
        # uMPs = float(uMPs[0])
        # # Collect the Unaveraged SCF Value
        # uSCF = [float(s[1]) for s in node_val if "Unavg SCF" in s[0]]
        # uSCF = float(uSCF[0])
        # # print((time_ref + 'Averaged SCF = %.2f | Unaveraged SCF = %.2f') % (time.time() - time_start,SCF,uSCF))
        
        # # Quality Control Point
        # # If the values disagree past a limit, then raise a flag
        # scf_limit = 0.1
        # scf_err = abs(uSCF - SCF)/uSCF
        # if scf_err >= scf_limit:
        #     # print((time_ref + 'Error: the comparison between the average and unaveraged SCF values exceeds 10%%') % (time.time() - time_start))
        #     sys.exit('The comparison between the average and unaveraged SCF values exceeds 10%. Review the dent Abaqus .odb file for more information.')
            
        # # print((time_ref + 'Saved contents to scf_values.xlsx') % (time.time() - time_start))
        
        # # return SCF, MPs

    