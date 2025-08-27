# -*- coding: utf-8 -*-
"""
Individual Rainflow Analysis

@author: evalencia
"""

import numpy as np
import pandas as pd
import rainflow
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

press_hist_path = 'Pressure_History (no spike).xlsx'
results_path = 'rainflow_results'
results_path = os.path.join(os.getcwd(), results_path) + '\\'
os.mkdir(results_path)

OD = 6.95
WT = 0.250
SMYS = 42000
service_years = (6/365)
M = 3
min_range = 5

to_unit = 'psi'
time_col = 0
P_cols = [4]
press_hist_skiprows = None
press_hist_header   = 0

# press_hist_path = press_hist_path.replace('\\', '/')
# results_path = results_path.replace('\\','/')

df_segment_dents = [OD, WT, SMYS, service_years, M, min_range]

# =============================================================================
# FUNCTIONS
# =============================================================================

def unit_conversion(input_data, to_unit):
    if to_unit.lower() == 'kpa':
        # Convert from kPa to psi
        output_data = input_data * 0.145038
    elif to_unit.lower() == 'psi':
        # Keep data normal
        output_data = input_data
    return output_data

def Px(Lx,hx,P1,P2,K,L1,L2,h1,h2,D1,D2):
    """
    Taken from API 1176
    
    Note: the version in API 1183 Section 6.6.3.1 Rainflow Counting Equation (5) is incorrect.
    
    Parameters
    ----------
    Lx : float
        the location of point analysis, ft
    hx : float
        the elevation of point analysis, ft
    P1 : array
        the upstream discharge pressure, psig
    P2 : array
        the downstream suction pressure, psig
    K : float
        SG x (0.433 psi/ft), where SG = specific gravity of product
    L1 : float
        the location of upstream discharge station, ft
    L2 : float
        the location of downstream suction station, ft
    h1 : float
        the elevation of upstream discharge station, ft
    h2 : float
        the elevation of downstream suction station, ft
    D1 : float
        the pipe diameter of segment between L1 and Lx, in
    D2 : float
        the pipe diameter of segment between Lx and L2, in

    Returns
    -------
    The intermediate pressure point between pressure sources, psig

    """

    Px = (P1 + K*h1 - P2 - K*h2)*(1/(((Lx - L1)*D2**5)/((L2 - Lx)*D1**5) + 1)) - K*(hx - h2) + P2
    
    return Px

def MD49(cycles, OD, WT, SMYS, service_years, M, min_range=5):
    # Create an empty array for all the MD-4-9 bins
    MD49_bins = np.zeros(28)
    MD49_P_range = np.array([10,20,30,40,50,60,70,
                             10,20,30,40,50,60,
                             10,20,30,40,50,
                             10,20,30,40,
                             10,20,30,
                             10,20,
                             10])
    
    # Remove any MD49_cycles that have a pressure range below the minimum range
    # MD49_cycles[:,2] = [0 for press_range in MD49_cycles[:,0] if press_range < min_range]
    MD49_cycles = cycles.copy()
    for i, val in enumerate(MD49_cycles[:,0]):
        if MD49_cycles[i,0] < min_range: 
            MD49_cycles[i,2] = 0
    
    # Reference Stress Ranges
    SSI_ref_stress = 13000  # psi
    
    # Convert pressure range into units of % SMYS
    MD49_cycles[:,0] = 100*MD49_cycles[:,0]*OD/(2*WT)/SMYS
    MD49_cycles[:,1] = 100*MD49_cycles[:,1]*OD/(2*WT)/SMYS
    
    # Iterate through every pressure range cycle
    for i, press_range in enumerate(MD49_cycles[:, 0]):
        # Pressure range: 0 - 10% SMYS
        if MD49_cycles[i, 0] <= 10.0: #if range is 0 - 10% SMYS
            if MD49_cycles[i, 1] <= 20.0: #if mean is 0 - 20% SMYS
                MD49_bins[0] = MD49_bins[0] + MD49_cycles[i, 2] #BIN #1
            elif MD49_cycles[i, 1] <= 30.0: #if mean is 20 - 30% SMYS
                MD49_bins[7] = MD49_bins[7] + MD49_cycles[i, 2] #BIN #8
            elif MD49_cycles[i, 1] <= 40.0: #if mean is 30 - 40% SMYS
                MD49_bins[13] = MD49_bins[13] + MD49_cycles[i, 2] #BIN #14
            elif MD49_cycles[i, 1] <= 50.0: #if mean is 40 - 50% SMYS
                MD49_bins[18] = MD49_bins[18] + MD49_cycles[i, 2] #BIN #19
            elif MD49_cycles[i, 1] <= 60.0: #if mean is 50 - 60% SMYS
                MD49_bins[22] = MD49_bins[22] + MD49_cycles[i, 2] #BIN #23
            elif MD49_cycles[i, 1] <= 70.0: #if mean is 60 - 70% SMYS
                MD49_bins[25] = MD49_bins[25] + MD49_cycles[i, 2] #BIN #26
            else: #if mean is >70% SMYS
                MD49_bins[27] = MD49_bins[27] + MD49_cycles[i, 2] #BIN #28
        # Pressure range: 10 - 20% SMYS
        elif MD49_cycles[i, 0] <= 20.0: #if range is 10 - 20% SMYS
            if MD49_cycles[i, 1] <= 25.0: #if mean is 0 - 25% SMYS
                MD49_bins[1] = MD49_bins[1] + MD49_cycles[i, 2] #BIN #2
            elif MD49_cycles[i, 1] <= 35.0: #if mean is 25 - 35% SMYS
                MD49_bins[8] = MD49_bins[8] + MD49_cycles[i, 2] #BIN #9
            elif MD49_cycles[i, 1] <= 45.0: #if mean is 35 - 45% SMYS
                MD49_bins[14] = MD49_bins[14] + MD49_cycles[i, 2] #BIN #15
            elif MD49_cycles[i, 1] <= 55.0: #if mean is 45 - 55% SMYS
                MD49_bins[19] = MD49_bins[19] + MD49_cycles[i, 2] #BIN #20
            elif MD49_cycles[i, 1] <= 65.0: #if mean is 55 - 65% SMYS
                MD49_bins[23] = MD49_bins[23] + MD49_cycles[i, 2] #BIN #24
            else: #if mean is >65% SMYS
                MD49_bins[26] = MD49_bins[26] + MD49_cycles[i, 2] #BIN #27
        # Pressure range: 20 - 30% SMYS
        elif MD49_cycles[i, 0] <= 30.0: #if range is 20 - 30% SMYS
            if MD49_cycles[i, 1] <= 30.0: #if mean is 0 - 30% SMYS
                MD49_bins[2] = MD49_bins[2] + MD49_cycles[i, 2] #BIN #3
            elif MD49_cycles[i, 1] <= 40.0: #if mean is 30 - 40% SMYS
                MD49_bins[9] = MD49_bins[9] + MD49_cycles[i, 2] #BIN #10
            elif MD49_cycles[i, 1] <= 50.0: #if mean is 40 - 50% SMYS
                MD49_bins[15] = MD49_bins[15] + MD49_cycles[i, 2] #BIN #16
            elif MD49_cycles[i, 1] <= 60.0: #if mean is 50 - 60% SMYS
                MD49_bins[20] = MD49_bins[20] + MD49_cycles[i, 2] #BIN #21
            else: #if mean is >60% SMYS
                MD49_bins[24] = MD49_bins[24] + MD49_cycles[i, 2] #BIN #25
        # Pressure range: 30 - 40% SMYS
        elif MD49_cycles[i, 0] <= 40.0: #if range is 30 - 40% SMYS
            if MD49_cycles[i, 1] <= 35.0: #if mean is 0 - 35% SMYS
                MD49_bins[3] = MD49_bins[3] + MD49_cycles[i, 2] #BIN #4
            elif MD49_cycles[i, 1] <= 45.0: #if mean is 35 - 45% SMYS
                MD49_bins[10] = MD49_bins[10] + MD49_cycles[i, 2] #BIN #11
            elif MD49_cycles[i, 1] <= 55.0: #if mean is 45 - 55% SMYS
                MD49_bins[16] = MD49_bins[16] + MD49_cycles[i, 2] #BIN #17
            else: #if mean is >55% SMYS
                MD49_bins[21] = MD49_bins[21] + MD49_cycles[i, 2] #BIN #22
        # Pressure range: 40 - 50% SMYS
        elif MD49_cycles[i, 0] <= 50.0: #if range is 40 - 50% SMYS
            if MD49_cycles[i, 1] <= 40.0: #if mean is 0 - 40% SMYS
                MD49_bins[4] = MD49_bins[4] + MD49_cycles[i, 2] #BIN #5
            elif MD49_cycles[i, 1] <= 50.0: #if mean is 40 - 50% SMYS
                MD49_bins[11] = MD49_bins[11] + MD49_cycles[i, 2] #BIN #12
            else: #if mean is >50% SMYS
                MD49_bins[17] = MD49_bins[17] + MD49_cycles[i, 2] #BIN #18
        # Pressure range: 50 - 60% SMYS
        elif MD49_cycles[i, 0] <= 60.0: #if range is 50 - 60% SMYS
            if MD49_cycles[i, 1] <= 45.0: #if mean is 0 - 45% SMYS
                MD49_bins[5] = MD49_bins[5] + MD49_cycles[i, 2] #BIN #6
            else: #if mean is >45% SMYS
                MD49_bins[12] = MD49_bins[12] + MD49_cycles[i, 2] #BIN #13
        # Pressure range > 60% SMYS
        else: #if pressure range > 60% SMYS
            MD49_bins[6] = MD49_bins[6] + MD49_cycles[i, 2] #BIN #7
            
    # Calculate the MD49 equivalent cycles
    MD49_final_cycles = sum(((((MD49_P_range/100)*SMYS)/SSI_ref_stress)**M)*MD49_bins)/service_years
    
    return MD49_final_cycles, MD49_bins

def equivalent_cycles(index, cycles, OD, WT, SMYS, service_years, M, min_range=5):
    """
    Parameters
    ----------
    index : string
        SSI or CI. SSI is the Spectrum Severity Indicator, CI is the Cyclic Index
    cycles : array of floats
        the array output from the rainflow analysis
    service_years : float
        the period of time in years for the pressure history data
    min_range : array
        the threshold value for pressure ranges to consider. Default is 5 psi.

    Returns
    -------
    Either the SSI or CI. 

    """
    equiv_cycles = np.zeros(cycles.shape[0])
    
    # Reference Stress Ranges
    SSI_ref_stress = 13000  # psi
    SSI_ref_press = SSI_ref_stress*2*WT/OD
    CI_ref_stress = 37580   # psi
    CI_ref_press = CI_ref_stress*2*WT/OD
    
    if index.lower() == 'ssi':
        ref_press = SSI_ref_press
    elif index.lower() == 'ci':
        ref_press = CI_ref_press
        
    for i, val in enumerate(cycles[:,0]):
        if cycles[i,0] > min_range: 
            equiv_cycles[i] = ((cycles[i,0]/ref_press)**M)*cycles[i,2]
        else:
            equiv_cycles[i] = 0
            
    num_cycles = sum(equiv_cycles)/service_years
    
    return num_cycles

def gas(press_hist_path, results_path, P_cols, press_hist_header, press_hist_skiprows, df_segment_dents, time_col=0, to_unit='psi'):
    """
    Compare the pressure history for every station and select the pressure history 
    that results in the most conservative pressure history (i.e., highest SSI).
    
    Parameters
    ----------
    press_hist_path : string
        the path to the pressure history data file.
    P_cols : array of index
        the location of various pressure stations in the flow direction (upstream to downstream).
    press_hist_header : index
        the data header index. If there is no data header, use None.
    press_hist_skiprows : index
        the number of data rows to skip when importing the data.
    df_segment_dents : DataFrame
        DataFrame containing a single dent to calculate the SSI as a reference point for pressure history comparison.
    time_col : index
        the location of the time stamp in the pressure history data file.

    Returns
    -------
    The SSI, CI, and MD49_SSI.
    """
    
    # Extract the information for the specified dent
    # dent_ID, pipe_segment, odometer, OD, WT, SMYS, service_years, min_range, M, fatigue_curve, dent_depth, dent_depth_in, dent_length, dent_width, dent_orientation, Lx, hx, SG, K, L1, L2, h1, h2, D1, D2, flag = df_segment_dents
    OD, WT, SMYS, service_years, M, min_range = df_segment_dents
    
    # Select the file for analysis. Supported files are:
    #   csv, xls, xlsx, xlsm, xlsb, odf, ods, and odt

    # file_path = filedialog.askopenfilename()
    filetypes = ('.xls', '.xlsx', '.xlsm', '.xlsb', '.odf', '.ods', '.odt')
    # pd.set_option('display.max_columns', None)

    if press_hist_path.endswith(filetypes):
        df = pd.read_excel(press_hist_path, index_col=None, header=press_hist_header, skiprows=press_hist_skiprows)
    elif press_hist_path.endswith('.csv'):
        df = pd.read_csv(press_hist_path, index_col=None, header=press_hist_header, skiprows=press_hist_skiprows)
    
    # Treat it as individual data. Remove any values that has either no value or NAN in the column
    P_list = []
    cycles_list = []
    P_time_list = []
    SSI_list = []
    CI_list = []
    MD49_SSI_list = []
    
    # Use the pressure history for every station and compare the end results to find the most conservative value
    for i in range(len(P_cols)):    
        
        # Make a copy of the dataframe, convert all values to numeric (if error, NaN), then drop NaNs
        df_ph = df.copy()
        df_ph[df_ph.columns[P_cols[i]]] = pd.to_numeric(df_ph[df_ph.columns[P_cols[i]]])#, errors='coerce')
        df_ph = df_ph.dropna()
        
        # Save the Pressure History and Time. Use this to perform rainflow analysis
        P_list.append(unit_conversion(df_ph.iloc[:,P_cols[i]], to_unit))
        P_time_list.append(df_ph.iloc[:,time_col])
        
        # Rainflow Analysis using Python package 'rainflow'
        cycles_list.append(pd.DataFrame(rainflow.extract_cycles(P_list[i])).to_numpy())
        
        # Calculate the SSI, CI, and SSI MD49
        SSI_list.append(equivalent_cycles('ssi', cycles_list[i], OD, WT, SMYS, service_years, M, min_range))
        CI_list.append(equivalent_cycles('ci', cycles_list[i], OD, WT, SMYS, service_years, M, min_range=5))
        MD49_SSI_list.append(MD49(cycles_list[i], OD, WT, SMYS, service_years, M, min_range))
    
    # Graphing
    gas_graphing(P_list, P_time_list, df.columns[1:], results_path)
    
    # return max_index, SSI, CI, MD49_SSI, cycles, MD49_bins, P_list
    return SSI_list, CI_list, MD49_SSI_list, cycles_list, P_list, df.columns[1:]

def gas_graphing(P_list, P_time, header_names, results_path):
    # Save the figures to the dent results folder to observe all of the pressure histories
    for i in range(len(P_list)):
        fig, sp = plt.subplots(figsize=(8,4), dpi=240)
        fig.suptitle('Pressure History for %s'%(header_names[i]), fontsize=16)
        sp.scatter(P_time[i], P_list[i], s=0.1)
        sp.set_ylabel('Pressure (psig)')
        sp.set_xlabel('Start Date Time')
        fig.savefig(results_path + 'P' + str(i + 1) + '_pressure_history.png')
    
    # If there are multiple pressure histories, then save one combined version
    if len(P_list) > 1:
        fig, sp = plt.subplots(figsize=(8,4), dpi=240)
        fig.suptitle('Combined Pressure History', fontsize=16)
        sp.set_ylabel('Pressure (psig)')
        sp.set_xlabel('Start Date Time')
        for i in range(len(P_list)):
            sp.scatter(P_time[i], P_list[i], s=0.1, label=header_names[i])
        
        sp.legend()
        fig.savefig(results_path + 'ALL_pressure_history.png')
            
# =============================================================================
# EXECUTE
# =============================================================================

SSI_list, CI_list, MD49_SSI_list, cycles_list, P_list, header_names = gas(press_hist_path, results_path, P_cols, press_hist_header, press_hist_skiprows, df_segment_dents, time_col, to_unit)

# Find the most conservative result (most conservative = highest SSI)
max_index = np.argmax(SSI_list)
SSI = max(SSI_list)
CI = CI_list[SSI_list.index(SSI)]
MD49_SSI = MD49_SSI_list[SSI_list.index(SSI)][0]
cycles = cycles_list[SSI_list.index(SSI)]
MD49_bins = MD49_SSI_list[SSI_list.index(SSI)][1]

# # Save the MD49_bins and cycles to a .txt file
# np.savetxt('md49_bins.csv',MD49_bins,delimiter=',')
# np.savetxt('cycles.csv', cycles, delimiter=',')

# Print the summary of results
parameters = ['SSI','CI','MD49']
df_summary = pd.DataFrame(index=parameters, columns=header_names)

# Save all of the rainflow cycles and md49_bins outputs per pressure history
for i in range(len(P_list)):
    # Rename the column to include pressure index (for quick reference)
    # df_summary.columns[i] = 'P' + str(i + 1) + ': ' + str(df_summary.columns[i])
    df_summary = df_summary.rename(columns={df_summary.columns[i]: 'P' + str(i + 1) + ': ' + str(df_summary.columns[i])})
    df_summary[df_summary.columns[i]] = [SSI_list[i], CI_list[i], MD49_SSI_list[i][0]]
    
    np.savetxt(results_path + 'P' + str(i + 1) + '_cycles.csv', cycles_list[i], delimiter=',')
    np.savetxt(results_path + 'P' + str(i + 1) + '_md49_bins.csv', MD49_SSI_list[i][1], delimiter=',')
    
# Print results on Console
w = 'Rainflow results for %.3f-inch OD x %.3f-inch WT, %i Grade, with data for %.3f year(s), using m = %.3f and min_range = %.3f-psig'% (OD, WT, SMYS, service_years, M, min_range)
w = w + '\n' + tabulate(df_summary, headers='keys', tablefmt='psql')
w = w + '\n' + 'Maximum pressure history found in %s.'%(header_names[max_index])
print(w)
# print('Rainflow results for %.3f-inch OD x %.3f-inch WT, %i Grade, with data for %.3f year(s), using m = %.3f and min_range = %.3f-psig'%
#       (OD, WT, SMYS, service_years, M, min_range))
# print(tabulate(df_summary, headers='keys', tablefmt='psql'))
# print('Maximum pressure history found in %s.'%(header_names[max_index]))

# Export results as .txt
with open(results_path + '/results.txt', 'w') as f:
    f.writelines(w)
    f.close()