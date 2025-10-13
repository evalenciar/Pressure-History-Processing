"""
Custom VBA Functions converted into Python
VBA Functions cover MD-2-4 and MD-5-3 methods for determining the Stress Range Magnification Factor (Km)
"""

import math
import json
import os
import openpyxl
import numpy as np
import pandas as pd

import rainflow_analysis as rfa
import md49
import traceback

tables_folder = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\Coefficient Tables"
rla_template = r"C:\Users\emman\Documents\SoftDev\Pressure History Processing\templates\RLA (v1.8.3) Display Only.xlsx"

table_md = 'Tables_Coefficients.json'
table_fatigue = 'Tables_FatigueCurves.json'

def get_km_l0(od: float, wt: float, restraint: str, max_pct_psmys: float) -> float | str:
    if od <= 0 or wt <= 0:
        return "Error in Inputs"

    if max_pct_psmys <= 1:
        max_pct_psmys *= 100

    r = od / wt

    t = restraint.replace("-", " ").replace(chr(8211), " ").strip().upper()

    if t == "UNRESTRAINED":
        if max_pct_psmys <= 20:
            km = 9.4 * (1 - math.exp(-0.045 * r))  # (A1)
        else:
            km = 7.5 * (1 - math.exp(-0.065 * r))  # (A2)
    elif t == "SHALLOW RESTRAINED":
        km = 0.1183 * r - 1.146  # (A3)
    elif t == "DEEP RESTRAINED":
        km = 0.1071 * r + 0.1332  # (A4)
    elif t == "RESTRAINED":
        km_sh = 0.1183 * r - 1.146
        km_dp = 0.1071 * r + 0.1332
        km = max(km_sh, km_dp)
    elif "MIXED" in t:
        if max_pct_psmys <= 20:
            result_unres = 9.4 * (1 - math.exp(-0.045 * r))  # (A1)
        else:
            result_unres = 7.5 * (1 - math.exp(-0.065 * r))  # (A2)
        if "SHALLOW" in t:
            result_res = 0.1183 * r - 1.146  # (A3)
        elif "DEEP" in t:
            result_res = 0.1071 * r + 0.1332  # (A4)
        else:
            return "Error"  # Handle case where mixed but no shallow/deep specified
        km = max(result_unres, result_res)
    else:
        return "Error"

    return km

def get_a5(OD: float, t: float, dP: float, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5 Unrestrained
    Coefficients from Table A.2
    """
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    # Assuming the structure: data["A.2"][0] contains the dict with keys like "a00", "a10", etc.
    coeffs = data["A.2"][0]

    x = dP / 100
    y = OD / t

    result = (
        coeffs["a00"] +
        coeffs["a10"] * x +
        coeffs["a01"] * y +
        coeffs["a20"] * x ** 2 +
        coeffs["a11"] * x * y +
        coeffs["a02"] * y ** 2 +
        coeffs["a30"] * x ** 3 +
        coeffs["a21"] * x ** 2 * y +
        coeffs["a12"] * x * y ** 2
    )

    return result

def get_a6(OD: float, t: float, dP: float, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5 Shallow Restrained
    Coefficients from Table A.3
    """
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    # Assuming the structure: data["A.3"][0] contains the dict with keys like "a0", "a1", etc.
    coeffs = data["A.3"][0]

    x = dP / 100
    y = OD / t

    poly_part = (
        coeffs["a0"] +
        coeffs["a1"] * x +
        coeffs["a2"] * y +
        coeffs["a3"] * x * y +
        coeffs["a4"] * x ** 2 +
        coeffs["a5"] * y ** 2 +
        coeffs["a6"] * x ** 3 +
        coeffs["a7"] * x ** 2 * y +
        coeffs["a8"] * x * y ** 2 +
        coeffs["a9"] * y ** 3
    )

    exp_part = (
        math.exp(-abs(coeffs["a10"] * x)) +
        math.exp(-abs(coeffs["a11"] * y + coeffs["a12"] * y ** 2 + coeffs["a13"] * y ** 3))
    )

    result = poly_part * exp_part

    return result

def get_a7(OD: float, t: float, dP: float, tables_path: str = os.path.join(tables_folder, table_md)) -> float:
    """
    Level 0.5 Deep Restrained
    Coefficients from Table A.4
    """
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)
    
    # Assuming the structure: data["A.4"][0] contains the dict with keys like "a1", "a2", etc.
    coeffs = data["A.4"][0]

    x = dP / 100
    y = OD / t

    result = (
        coeffs["a1"] +
        coeffs["a2"] * x +
        coeffs["a3"] * y +
        coeffs["a4"] * x * y +
        coeffs["a5"] * x ** 2 +
        coeffs["a6"] * y ** 2 +
        coeffs["a7"] * x ** 2 * y +
        coeffs["a8"] * x * y ** 2 +
        coeffs["a9"] * x ** 3 +
        coeffs["a10"] * y ** 3
    )

    return result

def get_km_l05(restraint: str, OD: float, WT: float, dP: float) -> float | str:
    # Check for empty or invalid inputs
    if not all([restraint, OD, WT, dP]):
        return "Error"

    restraint = restraint.strip().lower()

    if restraint == "unrestrained":
        return get_a5(OD, WT, dP)
    elif restraint == "shallow restrained":
        return get_a6(OD, WT, dP)
    elif restraint == "deep restrained":
        return get_a7(OD, WT, dP)
    elif "mixed" in restraint:
        result_unres = get_a5(OD, WT, dP)
        result_res = 0
        if "shallow" in restraint:
            result_res = get_a6(OD, WT, dP)
        elif "deep" in restraint:
            result_res = get_a7(OD, WT, dP)
        return max(result_unres, result_res)
    else:
        return "Error"

def get_a8(OD: float, t: float, dP: float, Pmean: float, restraint_condition: str, tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Level 0.5+
    Coefficients from Table A.5
    """
    # Load coefficients from JSON
    with open(tables_path, 'r') as f:
        data = json.load(f)

    table = data["A.5"]

    x = dP / 100
    y = Pmean / 100
    z = OD / t
    odt = OD / t

    # Normalize restraint_condition for matching
    restraint_condition = restraint_condition.strip().title()

    # Find the matching row
    coeffs = None
    for row in table:
        if row["Restraint"].strip().title() == restraint_condition and row["ODot_LB"] <= odt < row["ODot_UB"]:
            coeffs = row
            break

    if not coeffs:
        return "Out of Range"

    # Extract a1 to a14
    a = [coeffs[f"a{i}"] for i in range(1, 15)]

    # Calculate parts
    part1 = a[0] + a[1] * x + a[2] * y + a[3] * x * y + a[4] * x ** 2 + a[5] * y ** 2
    part2 = a[6] + a[7] * z + a[8] * z ** 2
    part3 = a[9] + a[10] * x + a[11] * y + a[12] * x ** 2 + a[13] * y ** 2

    result = part1 * part2 * math.exp(-abs(part3))

    return result

def get_km_l05p(OD: float, WT: float, dP: float, Pmean: float, restraint: str) -> float | str:
    # Check for empty or invalid inputs
    if not all([OD, WT, dP, Pmean, restraint]):
        return "Error"

    restraint = restraint.strip().lower()

    if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
        return get_a8(OD, WT, dP, Pmean, restraint.title())
    elif "mixed" in restraint:
        result_unres = get_a8(OD, WT, dP, Pmean, "Unrestrained")
        result_res = 0
        if "shallow" in restraint:
            result_res = get_a8(OD, WT, dP, Pmean, "Shallow Restrained")
        elif "deep" in restraint:
            result_res = get_a8(OD, WT, dP, Pmean, "Deep Restrained")
        return max(result_unres, result_res)
    else:
        return "Error"

def get_a9(OD: float, t: float, dP: float, depth: float, restraint_condition: str, tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Level 0.75
    Coefficients from Table A.6
    """
    try:
        # Load coefficients from JSON
        with open(tables_path, 'r') as f:
            data = json.load(f)

        table = data["A.6"]

        x = dP / 100
        y = OD / (t * 100)
        odt = OD / t
        d = depth

        # Determine which row to use based on restraint condition and OD/t ratio
        # Restraint Condition can be either "Restrained" or "Unrestrained"
        restraint_condition = "Unrestrained" if "unrestrained" in restraint_condition.lower() else "Restrained"
        restraint_condition = restraint_condition.strip().title()
        
        # Find the matching row
        coeffs = None
        for row in table:
            if (row["Restraint"].strip().title() == restraint_condition and 
                row["ODot_LB"] <= odt < row["ODot_UB"]):
                coeffs = row
                break

        if not coeffs:
            return "Out of Range"

        # Extract b1 to b15 coefficients
        b = [coeffs[f"b{i}"] for i in range(1, 16)]

        # Calculate intermediate values c1 through c5
        c1 = b[0] + (b[1] * x) + (b[2] * y)
        c2 = b[3] + (b[4] * x) + (b[5] * y)
        c3 = b[6] + (b[7] * x) + (b[8] * y)
        c4 = b[9] + (b[10] * x) + (b[11] * y)
        c5 = b[12] + (b[13] * x) + (b[14] * y)

        # Calculate final result
        result = abs(c1) - abs(c2) * ((abs(c3) - (abs(c4) * d) ** 2) * math.exp(-abs(c5) * d))

        return result
    except Exception:
        return "Error"

def get_km_l075(od: float, wt: float, dp: float, depth: float, restraint: str) -> float | str:
    """
    Level 0.75
    Return the max Km value
    """
    try:
        # Check for empty or invalid inputs
        if not all([od, wt, dp, depth, restraint]):
            return "Error"

        restraint = restraint.strip().lower()

        if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
            return get_a9(od, wt, dp, depth, restraint.title())
        elif "mixed" in restraint:
            result_unres = get_a9(od, wt, dp, depth, "Unrestrained")
            result_res = 0
            
            if "shallow" in restraint:
                result_res = get_a9(od, wt, dp, depth, "Shallow Restrained")
            elif "deep" in restraint:
                result_res = get_a9(od, wt, dp, depth, "Deep Restrained")
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                return "Error"
                
            return max(result_unres, result_res)
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def get_a10(od: float, t: float, dp: float, pmean: float, depth: float, restraint_condition: str, tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Level 0.75+
    Coefficients from Table A.7
    """
    try:
        # Load coefficients from JSON
        with open(tables_path, 'r') as f:
            data = json.load(f)

        table = data["A.7"]

        x = dp / 100
        y = pmean / 100
        z = od / (t * 100)
        odt = od / t
        d = depth

        # Determine which row to use based on restraint condition and OD/t ratio
        # Restraint Condition can be either "Restrained" or "Unrestrained"
        restraint_condition = "Unrestrained" if "unrestrained" in restraint_condition.lower() else "Restrained"
        restraint_condition = restraint_condition.strip().title()
        
        # Find the matching row
        coeffs = None
        for row in table:
            if (row["Restraint"].strip().title() == restraint_condition and 
                row["ODot_LB"] <= odt < row["ODot_UB"]):
                coeffs = row
                break

        if not coeffs:
            return "Out of Range"

        # Extract b1 to b20 coefficients
        b = [coeffs[f"b{i}"] for i in range(1, 21)]

        # Calculate intermediate values c1 through c5
        c1 = b[0] + b[1] * x + b[2] * y + b[3] * z
        c2 = b[4] + b[5] * x + b[6] * y + b[7] * z
        c3 = b[8] + b[9] * x + b[10] * y + b[11] * z
        c4 = b[12] + b[13] * x + b[14] * y + b[15] * z
        c5 = b[16] + b[17] * x + b[18] * y + b[19] * z

        # Calculate final result
        result = abs(c1) - abs(c2) * ((abs(c3) - (abs(c4) * d) ** 2) * math.exp(-abs(c5) * d))

        return result

    except Exception:
        return "Out of Range"

def get_km_l075p(od: float, wt: float, dp: float, pmean: float, depth: float, restraint: str) -> float | str:
    """
    Level 0.75+ - wrapper function for get_a10
    Return the max Km value
    """
    try:
        # Check for empty or invalid inputs
        if not all([od, wt, dp, pmean, depth, restraint]):
            return "Error"

        restraint = restraint.strip().lower()

        if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
            return get_a10(od, wt, dp, pmean, depth, restraint.title())
        elif "mixed" in restraint:
            result_unres = get_a10(od, wt, dp, pmean, depth, "Unrestrained")
            result_res = 0
            
            if "shallow" in restraint:
                result_res = get_a10(od, wt, dp, pmean, depth, "Shallow Restrained")
            elif "deep" in restraint:
                result_res = get_a10(od, wt, dp, pmean, depth, "Deep Restrained")
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                return "Error"
                
            return max(result_unres, result_res)
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def get_a16(od: float, t: float, pmean: float, pmax: float, pmin: float, pili: float, smys: float, 
           restraint_condition: str, lax_values: list, ltr_values: list, aax_values: list, atr_values: list,
           tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Level 1 & 2
    Uses coefficients from MD-5-3 Table A.8 and API 1183 Annex G Tables
    """
    try:
        # Load coefficients from JSON
        with open(tables_path, 'r') as f:
            data = json.load(f)

        # Calculate PF
        pf = (((pmax + pmin) * (pmax - pmin)) / (2 * 10000)) ** (1/3)
        
        # Calculate r
        r = -2.3053 * pf + 1.5685
        
        # Determine M based on restraint condition
        restraint_condition = restraint_condition.strip().title()
        if restraint_condition == "Unrestrained":
            m = 8
        else:
            m = 4
        
        # Calculate GSF
        gsf = (smys / 52) ** m
        
        # Convert ranges to lists (assuming they're already lists in Python)
        lax = lax_values
        ltr = ltr_values
        aax = aax_values
        atr = atr_values
        
        # Calculate xL and xH based on restraint condition
        if restraint_condition == "Deep Restrained":
            xl = ((aax[5] * aax[1]) ** 0.5 / (t * lax[3])) ** 1.5 * (lax[3] / ltr[3]) ** 0.5
            xh = (aax[8] / (lax[10] * lax[3])) ** 0.75 * (ltr[3] / lax[3])
            
        elif restraint_condition == "Shallow Restrained":
            xl = ((aax[5] * aax[1]) ** 0.5 / (t * lax[3])) ** 1.5 * (lax[3] / ltr[3]) ** 0.5
            xh = 10 * (ltr[10] / lax[9]) ** 0.5
            
        elif restraint_condition == "Unrestrained":
            # Get lambda values from G tables
            tmp = int(pmean/10) * 10  # Truncate the TMP value to floor
            rilip = round(pili/10) * 10  # Round RILIP value
            
            lambh = lambl = None
            # Search through G.1 to G.7 tables for matching values
            table = data["G.1_7"]
            for row in table:
                if row["TMP"] == tmp and row["RILIP"] == rilip:
                    lambh = row["lambH"]
                    lambl = row["lambL"]
                    break
            
            if lambh is None or lambl is None:
                return "Out of Range"
            
            xl = 10**4 * lambl * ((aax[0] * aax[1]) / (od * t**2 * lax[3])) ** 1.2 * (lax[2] / ltr[1]) ** 1.5
            xh = 10**4 * lambh * ((aax[1] * atr[1]) / (od * t * lax[3] * ltr[3])) ** 1.2 * (lax[3] / ltr[3]) ** 1.5
        else:
            return "Out of Range"
        
        # Calculate SP
        if restraint_condition == "Unrestrained":
            sp = (r * xl + (1 - r) * xh) * gsf
        else:
            sp = (r * xl + (1 - r) * xh) * gsf * (od / t) ** 0.25
        
        # Get coefficients A and B from Table A.8
        table_a8 = data["A.8"]
        
        a = b = None
        # Search through A.8 table for matching values
        for row in table_a8:
            if (row["Restraint"].strip().title() == restraint_condition and
                row["Pmin_%SMYS"] == pmin and row["Pmax_%SMYS"] == pmax):
                a = row["log10A"]
                b = row["B"]
                break

        if a is None or b is None:
            return "Out of Range"
        
        # Calculate final result
        result = (10 ** a) * sp ** b
        
        return result

    except Exception:
        return "Out of Range"
    
def get_a16_min(restraint_condition: str, od: float, t: float, pmean: float, pmax: float, pmin: float, 
               pili: float, smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
               ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
               uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
               dscw_atr_values: list) -> float | str:
    """
    Iterate through all four quadrants using the combinations of US-CCW, US-CW, DS-CCW, DS-CW 
    and return the minimum result from get_a16
    """
    try:
        # Check if any of the input lists are None or empty
        input_lists = [us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                      usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                      dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values]
        
        if any(lst is None or len(lst) == 0 for lst in input_lists):
            return "Error: Invalid input ranges"
        
        # Calculate results for all four quadrants
        results = []
        
        # Quadrant 1: US-CCW
        result1 = get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         us_lax_values, usccw_ltr_values, us_aax_values, usccw_atr_values)
        results.append(result1)
        
        # Quadrant 2: US-CW
        result2 = get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         us_lax_values, uscw_ltr_values, us_aax_values, uscw_atr_values)
        results.append(result2)
        
        # Quadrant 3: DS-CCW
        result3 = get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         ds_lax_values, dsccw_ltr_values, ds_aax_values, dsccw_atr_values)
        results.append(result3)
        
        # Quadrant 4: DS-CW
        result4 = get_a16(od, t, pmean, pmax, pmin, pili, smys, restraint_condition, 
                         ds_lax_values, dscw_ltr_values, ds_aax_values, dscw_atr_values)
        results.append(result4)
        
        # Filter out error results and find minimum
        valid_results = []
        for result in results:
            if isinstance(result, (int, float)) and not isinstance(result, str):
                valid_results.append(result)
        
        # Check if any valid results were found
        if valid_results:
            return min(valid_results)
        else:
            return "No valid results found"
    
    except Exception:
        return "Error in calculation"
    
def get_n_l2(restraint_condition: str, od: float, t: float, pmean: float, pmax: float, pmin: float, 
            pili: float, smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
            ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
            uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
            dscw_atr_values: list) -> float | str:
    """
    Returns the minimum Cycles to Failure for Level 2
    """
    try:
        # Check for empty or invalid restraint condition
        if not restraint_condition or restraint_condition.strip().lower() == "error":
            return "Error"

        restraint_condition = restraint_condition.strip().lower()

        if restraint_condition in ["unrestrained", "shallow restrained", "deep restrained"]:
            return get_a16_min(restraint_condition.title(), od, t, pmean, pmax, pmin, pili, smys,
                              us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                              usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                              dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
        
        elif "mixed" in restraint_condition:
            result_unres = get_a16_min("Unrestrained", od, t, pmean, pmax, pmin, pili, smys,
                                      us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                      usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                      dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            result_res = 0
            
            if "shallow" in restraint_condition:
                result_res = get_a16_min("Shallow Restrained", od, t, pmean, pmax, pmin, pili, smys,
                                        us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                        usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                        dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            elif "deep" in restraint_condition:
                result_res = get_a16_min("Deep Restrained", od, t, pmean, pmax, pmin, pili, smys,
                                        us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                        usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                        dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                return "Error"
                
            return min(result_unres, result_res)
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def get_a16_min_l1(restraint_condition: str, od: float, t: float, closest_pmean: float, pili: float, 
                      smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
                      ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
                      uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
                      dscw_atr_values: list) -> float | str:
    """
    Choose the applicable case, then iterate through the four quadrants using get_a16_min
    Level 1 specific function that maps ClosestPMean to Pmin/Pmax ranges
    """
    try:
        # Determine Pmin and Pmax based on ClosestPMean
        if closest_pmean == 25:
            pmin = 10
            pmax = 40
        elif closest_pmean == 45:
            pmin = 30
            pmax = 60
        elif closest_pmean == 65:
            pmin = 50
            pmax = 80
        else:
            return "Error: Invalid ClosestPMean value"
        
        # Call get_a16_min with the determined pressure values
        return get_a16_min(restraint_condition, od, t, closest_pmean, pmax, pmin, pili, smys,
                          us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                          usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                          dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)

    except Exception:
        return "Error in calculation"
    
def get_n_l1(restraint_condition: str, od: float, t: float, closest_pmean: float, pili: float, 
            smys: float, us_lax_values: list, us_aax_values: list, ds_lax_values: list, 
            ds_aax_values: list, usccw_ltr_values: list, usccw_atr_values: list, uscw_ltr_values: list, 
            uscw_atr_values: list, dsccw_ltr_values: list, dsccw_atr_values: list, dscw_ltr_values: list, 
            dscw_atr_values: list) -> float | str:
    """
    Returns the minimum Cycles to Failure for Level 1
    Note: SMYS is in units of ksi
    """
    try:
        # Check for empty or invalid restraint condition
        if not restraint_condition or restraint_condition.strip().lower() == "error":
            return "Error"

        restraint_condition = restraint_condition.strip().lower()

        if restraint_condition in ["unrestrained", "shallow restrained", "deep restrained"]:
            return get_a16_min_l1(restraint_condition.title(), od, t, closest_pmean, pili, smys,
                                 us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                 usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                 dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
        
        elif "mixed" in restraint_condition:
            result_unres = get_a16_min_l1("Unrestrained", od, t, closest_pmean, pili, smys,
                                         us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                         usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                         dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            result_res = 0
            
            if "shallow" in restraint_condition:
                result_res = get_a16_min_l1("Shallow Restrained", od, t, closest_pmean, pili, smys,
                                           us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                           usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                           dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            elif "deep" in restraint_condition:
                result_res = get_a16_min_l1("Deep Restrained", od, t, closest_pmean, pili, smys,
                                           us_lax_values, us_aax_values, ds_lax_values, ds_aax_values,
                                           usccw_ltr_values, usccw_atr_values, uscw_ltr_values, uscw_atr_values,
                                           dsccw_ltr_values, dsccw_atr_values, dscw_ltr_values, dscw_atr_values)
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                return "Error"
                
            return min(result_unres, result_res)
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def get_md24_unbinned(OD: float, WT: float, SMYS: float, dent_type: str, 
                     rainflow_data: list, ili_pressure: float, km_params: list, bs_sd: float = 0, fatigue_curves_path: str = os.path.join(tables_folder, table_fatigue)) -> dict | str:
    """
    MD-2-4 Unbinned damage calculation
    
    Args:
        OD: Outer diameter
        WT: Wall thickness
        SMYS: SMYS value
        dent_type: Type of dent
        rainflow_data: List of [pressure_range, pressure_mean, cycles] data
        ili_pressure: ILI Pressure as %SMYS
        km_params: List of 25 KM parameters in order:
                  [AH40, AH36, AH35, AJ38, AJ34, AJ33, AM40, AM36, AM35, AO38, AO34, AO33,
                   AM50, AO48, AM48, AH50, AJ48, AH48, AM64, AO62, AM62, AH64, AJ62, AH62]
                    # lax30_us, lax75_us, lax85_us,
                    # aax30_us, aax75_us, aax85_us,
                    # lax30_ds, lax75_ds, lax85_ds,
                    # aax30_ds, aax75_ds, aax85_ds,
                    # ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                    # ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                    # ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                    # ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
        bs_sd: Standard deviations for BS curve
        fatigue_curves_path: Path to the fatigue curves JSON file
    
    Returns:
        Dictionary of damage values organized by curve type and class, or error string
    """
    
    try:
        # Input validation
        if OD == 0 or WT == 0 or SMYS == 0:
            return "Check WT and/or SMYS."
        
        if len(rainflow_data) == 0:
            return "No rainflow data provided."
        
        if len(km_params) != 24:
            return "KM parameters list must contain exactly 24 values."

        # Load fatigue curve parameters from JSON
        with open(fatigue_curves_path, 'r') as f:
            fatigue_curves = json.load(f)
        
        # Extract KM parameters
        (lax30_us, lax75_us, lax85_us,
        aax30_us, aax75_us, aax85_us,
        lax30_ds, lax75_ds, lax85_ds,
        aax30_ds, aax75_ds, aax85_ds,
        ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
        ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
        ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
        ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw) = km_params
        # (ah40, ah36, ah35, aj38, aj34, aj33, am40, am36, am35, ao38, ao34, ao33,
        #  am50, ao48, am48, ah50, aj48, ah48, am64, ao62, am62, ah64, aj62, ah62, k62) = km_params
        
        # %SMYS conversion factor
        factor = 100.0 * (OD / (2.0 * WT)) / SMYS

        # Initialize damage dictionary
        damage_results = {
            "ABS": {},
            "DNV": {},
            "BS": {}
        }
        
        # Initialize damage for each curve class
        for curve_data in fatigue_curves["ABS"]:
            damage_results["ABS"][curve_data["Curve"]] = 0.0
            
        for curve_data in fatigue_curves["DNV"]:
            damage_results["DNV"][curve_data["Curve"]] = 0.0
            
        for curve_data in fatigue_curves["BS"]:
            damage_results["BS"][curve_data["Curve"]] = 0.0
        
        # Process each rainflow cycle
        for row in rainflow_data:
            if len(row) < 3:
                continue
                
            pr = float(row[0])  # Pressure Range (psi)
            pm = float(row[1])  # Pressure Mean (psi)
            cycles = float(row[2])  # Number of cycles
            
            # Convert to %SMYS
            prange_pct = pr * factor
            pmean_pct = pm * factor
            pmin_pct = (pm - 0.5 * pr) * factor
            pmax_pct = (pm + 0.5 * pr) * factor
            
            # Calculate Km using the get_km function (placeholder - you'll need to implement this)
            km = get_km(OD, WT, dent_type,
                        lax30_us, lax75_us, lax85_us,
                        aax30_us, aax75_us, aax85_us,
                        lax30_ds, lax75_ds, lax85_ds,
                        aax30_ds, aax75_ds, aax85_ds,
                        ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                        ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                        ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                        ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                        pmean_pct, ili_pressure, prange_pct / 100, pmean_pct / 100, pmax_pct / 100)
            
            if isinstance(km, str):  # Error in km calculation
                continue
                
            # Peak Stress Range (ksi)
            s_ksi = km * (prange_pct / 100) * SMYS / 1000
            
            # Convert to MPa for DNV and BS calculations
            s_mpa = s_ksi * 6.89476  # Convert from ksi to MPa (1 ksi = 6.89476 MPa)
            
            # ABS Damage Calculations
            for curve_data in fatigue_curves["ABS"]:
                curve_name = curve_data["Curve"]
                a_ksi = curve_data["A_ksi"]
                m = curve_data["m"]
                c_ksi = curve_data["C_ksi"]
                r = curve_data["r"]
                seq_ksi = curve_data["Seq_ksi"]
                
                if s_ksi < seq_ksi:
                    cycles_to_failure = c_ksi * (s_ksi ** (-r))
                else:
                    cycles_to_failure = a_ksi * (s_ksi ** (-m))
                
                if cycles_to_failure > 0:
                    damage_results["ABS"][curve_name] += cycles / cycles_to_failure
            
            # DNV Damage Calculations
            for curve_data in fatigue_curves["DNV"]:
                curve_name = curve_data["Curve"]
                m1 = curve_data["m1"]
                loga1 = curve_data["loga1"]
                m2 = curve_data["m2"]
                loga2 = curve_data["loga2"]
                seq_mpa = curve_data["Seq"]
                
                if s_mpa > seq_mpa:
                    exp_a = loga1 - m1 * math.log10(s_mpa)
                else:
                    exp_a = loga2 - m2 * math.log10(s_mpa)
                
                cycles_to_failure = 10 ** exp_a
                
                if cycles_to_failure > 0:
                    damage_results["DNV"][curve_name] += cycles / cycles_to_failure
            
            # BS Damage Calculations
            for curve_data in fatigue_curves["BS"]:
                curve_name = curve_data["Curve"]
                log10c0 = curve_data["log10C0"]
                m = curve_data["m"]
                sd = curve_data["SD"]
                soc = curve_data["Soc"]
                
                # Use d = 2 as standard deviation factor (common practice)
                d = bs_sd
                
                if s_mpa > soc:
                    cycles_to_failure = 10 ** (log10c0 - d * sd - m * math.log10(s_mpa))
                    
                    if cycles_to_failure > 0:
                        damage_results["BS"][curve_name] += cycles / cycles_to_failure
        
        return damage_results
        
    except Exception as e:
        traceback.print_exc()
        return f"Error in calculation: {str(e)}"
    
def get_km(od: float, wt: float, restraint_condition: str, 
           lax30_us: float, lax75_us: float, lax85_us: float,
           aax30_us: float, aax75_us: float, aax85_us: float,
           lax30_ds: float, lax75_ds: float, lax85_ds: float,
           aax30_ds: float, aax75_ds: float, aax85_ds: float,
           ltr75_us_cw: float, atr75_us_cw: float, ltr85_us_cw: float,
           ltr75_us_ccw: float, atr75_us_ccw: float, ltr85_us_ccw: float,
           ltr75_ds_cw: float, atr75_ds_cw: float, ltr85_ds_cw: float,
           ltr75_ds_ccw: float, atr75_ds_ccw: float, ltr85_ds_ccw: float,
           mean_pct_psmys_raw: float, ili_pct_psmys_raw: float,
           pr_over: float, pm_over: float, px_over: float,
           tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Compute per-side K_M (US/DS) and return the max
    """
    try:
        # Load tables from JSON
        with open(tables_path, 'r') as f:
            data = json.load(f)
        
        # Check for empty or invalid restraint condition
        if not restraint_condition or restraint_condition.strip().lower() == "error":
            return "Error"

        restraint_condition = restraint_condition.strip().lower()

        is_restrained = (restraint_condition != "unrestrained")

        # Deciles for lambda lookup
        mean_trunc = 10 * int(mean_pct_psmys_raw / 10)
        if mean_trunc < 10:
            mean_trunc = 10
        if mean_trunc > 70:
            mean_trunc = 70

        ili_round = 10 * int(ili_pct_psmys_raw / 10 + 0.5)
        if ili_round < 10:
            ili_round = 10
        if ili_round > 70:
            ili_round = 70

        # Lambda scale factors (1 if restrained)
        if is_restrained:
            lam1 = lam2 = 1.0
        else:
            # Get coefficients lambda1 and lambda2 from Table 9/10
            table = data["9_10"]

            lam1 = lam2 = None
            # Search through 9_10 table for matching values
            for row in table:
                if (row["TMP"] == mean_trunc and row["RILIP"] == ili_round):
                    lam1 = row["lamb1"]
                    lam2 = row["lamb2"]
                    break

            if lam1 is None or lam2 is None:
                return "Out of Range"
        
        # Aspect Ratios per side
        den85 = lax85_us + lax85_ds
        if den85 == 0:
            return "Error in Inputs"
        
        ar_us = (ltr85_us_cw + ltr85_us_ccw) / den85
        ar_ds = (ltr85_ds_cw + ltr85_ds_ccw) / den85
        
        # Per-side K_M calculation
        km_us = km_for_side(data, is_restrained, lam1, lam2, od, wt,
                           lax30_us, lax75_us, lax85_us, aax30_us, aax75_us, aax85_us,
                           ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                           ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                           ar_us, pr_over, pm_over, px_over)
        
        km_ds = km_for_side(data, is_restrained, lam1, lam2, od, wt,
                           lax30_ds, lax75_ds, lax85_ds, aax30_ds, aax75_ds, aax85_ds,
                           ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                           ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                           ar_ds, pr_over, pm_over, px_over)
        
        if km_us == -1e99 and km_ds == -1e99:
            return "Error in calculation"
        
        return max(km_us, km_ds)
        
    except Exception:
        return "Error in Inputs"
    
def get_km_l2(od: float, wt: float, restraint: str, 
            lax30_us: float, lax75_us: float, lax85_us: float,
            aax30_us: float, aax75_us: float, aax85_us: float,
            lax30_ds: float, lax75_ds: float, lax85_ds: float,
            aax30_ds: float, aax75_ds: float, aax85_ds: float,
            ltr75_us_cw: float, atr75_us_cw: float, ltr85_us_cw: float,
            ltr75_us_ccw: float, atr75_us_ccw: float, ltr85_us_ccw: float,
            ltr75_ds_cw: float, atr75_ds_cw: float, ltr85_ds_cw: float,
            ltr75_ds_ccw: float, atr75_ds_ccw: float, ltr85_ds_ccw: float,
            mean_pct_psmys_raw: float, ili_pct_psmys_raw: float,
            pr_over: float, pm_over: float, px_over: float,
            tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Wrapper for get_km to handle "Mixed" restraint condition for Level 2
    Returns the max Km value.
    """
    try:
        restraint = restraint.strip().lower()

        if restraint in ["unrestrained", "shallow restrained", "deep restrained"]:
            return get_km(od, wt, restraint, 
                            lax30_us, lax75_us, lax85_us,
                            aax30_us, aax75_us, aax85_us,
                            lax30_ds, lax75_ds, lax85_ds,
                            aax30_ds, aax75_ds, aax85_ds,
                            ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                            ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                            ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                            ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                            mean_pct_psmys_raw, ili_pct_psmys_raw,
                            pr_over, pm_over, px_over)
        elif "mixed" in restraint:
            result_unres = get_km(od, wt, "Unrestrained", 
                                lax30_us, lax75_us, lax85_us,
                                aax30_us, aax75_us, aax85_us,
                                lax30_ds, lax75_ds, lax85_ds,
                                aax30_ds, aax75_ds, aax85_ds,
                                ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                                ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                                ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                mean_pct_psmys_raw, ili_pct_psmys_raw,
                                pr_over, pm_over, px_over)
            result_res = 0
            
            if "shallow" in restraint:
                result_res = get_km(od, wt, "Shallow Restrained", 
                                lax30_us, lax75_us, lax85_us,
                                aax30_us, aax75_us, aax85_us,
                                lax30_ds, lax75_ds, lax85_ds,
                                aax30_ds, aax75_ds, aax85_ds,
                                ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                                ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                                ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                mean_pct_psmys_raw, ili_pct_psmys_raw,
                                pr_over, pm_over, px_over)
            elif "deep" in restraint:
                result_res = get_km(od, wt, "Deep Restrained", 
                                lax30_us, lax75_us, lax85_us,
                                aax30_us, aax75_us, aax85_us,
                                lax30_ds, lax75_ds, lax85_ds,
                                aax30_ds, aax75_ds, aax85_ds,
                                ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                                ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                                ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                mean_pct_psmys_raw, ili_pct_psmys_raw,
                                pr_over, pm_over, px_over)
            
            # Check if either result is an error string
            if isinstance(result_unres, str) or isinstance(result_res, str):
                return "Error"
                
            return max(result_unres, result_res)
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def km_for_side(data: dict, is_restrained: bool, lam1: float, lam2: float,
                od: float, wt: float, lax30: float, lax75: float, lax85: float,
                aax30: float, aax75: float, aax85: float,
                ltr75_cw: float, atr75_cw: float, ltr85_cw: float,
                ltr75_ccw: float, atr75_ccw: float, ltr85_ccw: float,
                ar_side: float, pr_over: float, pm_over: float, px_over: float) -> float:
    """
    One side (US or DS): compute AR bin, pull b's, evaluate CW & CCW, take max
    """
    try:
        # Determine the b coefficients based on restraint and AR bin
        b = None
        # Search through 9_10 table for matching values
        if is_restrained:
            table = data["29"]
        else:
            table = data["30"]
        for row in table:
            if (row["AR_LB"] <= ar_side < row["AR_UB"]):
                b = list(row.values())[2:]  # All b coefficients start from index 2
                break

        # Calculate KM for CW and CCW
        km_cw = km_for_pair(is_restrained, lam1, lam2, od, wt,
                           lax30, lax75, lax85, aax30, aax75, aax85,
                           ltr75_cw, atr75_cw, ltr85_cw,
                           pr_over, pm_over, px_over, b)
        
        km_ccw = km_for_pair(is_restrained, lam1, lam2, od, wt,
                            lax30, lax75, lax85, aax30, aax75, aax85,
                            ltr75_ccw, atr75_ccw, ltr85_ccw,
                            pr_over, pm_over, px_over, b)
        
        if km_cw == -1e99 and km_ccw == -1e99:
            return -1e99
        else:
            return max(km_cw, km_ccw)
            
    except Exception:
        return -1e99
    
def km_for_pair(is_restrained: bool, lam1: float, lam2: float,
                od: float, wt: float, lax30: float, lax75: float, lax85: float,
                aax30: float, aax75: float, aax85: float,
                ltr75: float, atr75: float, ltr85: float,
                pr_over: float, pm_over: float, px_over: float, b: list) -> float:
    """
    One pairing (e.g., CW) for a given side
    """
    try:
        if wt <= 0 or od <= 0:
            return -1e99
        
        if is_restrained:
            # Eq. (4): restrained
            if (lax30 <= 0 or lax75 <= 0 or ltr75 <= 0 or 
                aax30 <= 0 or aax75 <= 0 or atr75 <= 0):
                return -1e99
            
            x1 = (aax30 * aax75) / (lax30 * lax75 * od * wt * 0.001) * (ltr75 / lax75) ** 0.5
            x2 = ((aax75 * atr75) / (lax75 * ltr75 * wt ** 2)) ** 0.5
            xbar1 = (x1 - 2.97593) / 4.02113
            xbar2 = (x2 - 0.22786) / 0.16693
            
            return km_restrained_from_b(od, wt, xbar1, xbar2, pr_over, pm_over, px_over, b)
        else:
            # Eq. (8): unrestrained
            if (lax30 <= 0 or lax75 <= 0 or lax85 <= 0 or ltr85 <= 0 or
                aax30 <= 0 or aax75 <= 0 or aax85 <= 0):
                return -1e99
            
            x1 = lam1 * (math.sqrt(aax85 * aax75) / (lax85 * lax75)) * (lax85 / ltr85) ** 0.25
            x2 = lam2 * (aax30 / (lax30 * wt)) ** 0.25
            xbar1 = (x1 - 0.01478) / 0.014801
            xbar2 = (x2 - 0.67486) / 0.10759
            
            return km_unrestrained_from_b(od, wt, xbar1, xbar2, pr_over, pm_over, px_over, b)
            
    except Exception:
        return -1e99
    
def km_restrained_from_b(od: float, wt: float, xbar1: float, xbar2: float,
                        pr: float, pm: float, px: float, b: list) -> float:
    """
    KM assembly (Restrained, Eq. 2): product of two sums
    """
    try:
        geom = (od / wt) / 100
        c = build_c_from_b(b, True, geom, pr, pm, px)
        
        if not c or len(c) < 11:
            return -1e99
        
        sum_a = (abs(c[1]) + 
                abs(c[2]) * (xbar1 + c[3]) ** 2 + 
                abs(c[4]) * (xbar2 + c[5]) ** 2)
        
        sum_b = (abs(c[6]) + 
                abs(c[7]) * math.exp(-((xbar1 + c[8]) ** 2)) + 
                abs(c[9]) * math.exp(-((xbar2 + c[10]) ** 2)))
        
        return sum_a * sum_b
        
    except Exception:
        return -1e99

def km_unrestrained_from_b(od: float, wt: float, xbar1: float, xbar2: float,
                          pr: float, pm: float, px: float, b: list) -> float:
    """
    KM assembly (Unrestrained, Eq. 6): product of two sums
    """
    try:
        geom = (od / wt) / 100
        c = build_c_from_b(b, False, geom, pr, pm, px)
        
        if not c or len(c) < 11:
            return -1e99
        
        sum_a = (abs(c[1]) + 
                abs(c[2]) * (xbar1 + c[3]) ** 2 + 
                abs(c[4]) * (xbar2 + c[5]) ** 2)
        
        sum_b = (abs(c[6]) + 
                abs(c[7]) * math.exp(-((xbar1 + c[8]) ** 2)) + 
                abs(c[9]) * math.exp(-((xbar2 + c[10]) ** 2)))
        
        return sum_a * sum_b
        
    except Exception:
        return -1e99
    
def build_c_from_b(b: list, is_restrained: bool, geom: float, 
                   pr: float, pm: float, px: float) -> list:
    """
    Build c1..c10 from b (Table 29/30)
    """
    try:
        if not b:
            return []
        
        c = [0] * 11  # c[0] unused, c[1] to c[10]
        
        if is_restrained:
            # Table 29: 5 numbers per c_n (b1..b50)
            if len(b) < 50:
                return []
            
            for n in range(1, 11):
                k = 5 * (n - 1)
                c[n] = (b[k] + b[k+1] * geom + b[k+2] * pr + 
                       b[k+3] * pm + b[k+4] * px)
        else:
            # Table 30: 4 numbers per c_n (b1..b40)
            if len(b) < 40:
                return []
            
            for n in range(1, 11):
                k = 4 * (n - 1)
                c[n] = (b[k] + b[k+1] * geom + b[k+2] * pr + b[k+3] * pm)
        
        return c
        
    except Exception:
        return []
    
def get_scale_factor_md53(certainty: float, safety_factor: float, criteria: str, level: str, 
                          tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Get the scale factor for Level 0 through 2 using Table D.1 through D.7 from MD-5-3
    """
    with open(tables_path, 'r') as f:
        data = json.load(f)
    try:
        table = data["D.1_6"]

        sf = None
        # Search through D.1_6 table for matching values
        # If Criteria is "Multiple", treat it as "Corrosion"
        if criteria.strip().lower() == "multiple" or criteria.strip().lower() == "metal loss" or criteria.strip().lower() == "corrosion":
            criteria = "Corrosion"

        # Make sure that Certainty is a decimal value (e.g., 0.9, 0.8, 0.7)
        if certainty > 1.0:
            certainty = certainty / 100.0

        # If level is "1", treat it as "2"
        if level == "1":
            level = "2"

        for row in table:
            if (str(row["Level"]) == level and 
                row["Interaction"] == criteria and 
                row["Safety_Factor"] == safety_factor and
                row["Certainty"] == certainty):
                sf = row["Scale_Factor"]
                break

        if sf is None:
            return "Out of Range"
        
        return sf
    
    except Exception:
        return "Error in calculation"

def get_scale_factor_l2(restraint: str, certainty: float, safety_factor: float, 
                       criteria: str, metal_loss_location: str = math.nan, weld_interaction_sf: int = math.nan,
                       tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Get the scale factor for Level 2 only using Tables 15 through 18 from MD-2-4
    """
    with open(tables_path, 'r') as f:
        data = json.load(f)
    try:
        table = data["15_28"]

        sf = None
        # Search through 15_28 table for matching values
        # If Criteria is "Multiple", treat it as "Corrosion" and ignore weld_interaction_sf
        if criteria.strip().lower() == "multiple" or criteria.strip().lower() == "metal loss" or criteria.strip().lower() == "corrosion":
            criteria = "Corrosion"
            weld_interaction_sf = math.nan  # Ignore weld interaction SF

        if criteria.strip().lower() == "plain":
            weld_interaction_sf = math.nan  # Ignore weld interaction SF
            metal_loss_location = math.nan

        if criteria.strip().lower() == "weld":
            metal_loss_location = math.nan  # Ignore metal loss location

        # Make sure that Certainty is a decimal value (e.g., 0.9, 0.8, 0.7)
        if certainty > 1.0:
            certainty = certainty / 100.0

        # If metal_loss_location is not provided, set it to NaN
        if not metal_loss_location or metal_loss_location == "":
            metal_loss_location = math.nan

        # Check the Restraint input and convert it to either "Restrained" or "Unrestrained" based on substring
        # Treat Shallow/Deep Restrained as "Restrained"
        if "unrestrained" in restraint.strip().lower():
            restraint = "Unrestrained"
        else:
            restraint = "Restrained"

        for row in table:
            if (row["Interaction"] == criteria and 
                ((pd.isna(row["OD_ID"]) and pd.isna(metal_loss_location)) or (row["OD_ID"] == metal_loss_location)) and
                ((pd.isna(row["Reduction_Factor"]) and pd.isna(weld_interaction_sf)) or (row["Reduction_Factor"] == weld_interaction_sf)) and
                row["Restraint"] == restraint and
                row["Safety_Factor"] == safety_factor and
                row["Certainty"] == certainty):
                sf = row["Scale_Factor"]
                break

        if sf is None:
            return "Out of Range"
        
        return sf
    
    except Exception:
        traceback.print_exc()
        return "Error in calculation"
    
def get_scale_factor_l2_wrap(restraint: str, certainty: float, safety_factor: float, 
                            criteria: str, metal_loss_location: str = math.nan, weld_interaction_sf: int = math.nan,
                            tables_path: str = os.path.join(tables_folder, table_md)) -> float | str:
    """
    Wrapper function for get_scale_factor_l2 to repeat calculations if Restraint is "Mixed"
    Return the maximum scale factor from both restraint conditions
    Treat Shallow/Deep Restrained as "Restrained"
    """
    if "mixed" in restraint.strip().lower():
        sf1 = get_scale_factor_l2("Restrained", certainty, safety_factor, criteria, metal_loss_location, weld_interaction_sf, tables_path)
        sf2 = get_scale_factor_l2("Unrestrained", certainty, safety_factor, criteria, metal_loss_location, weld_interaction_sf, tables_path)
        return max(sf1, sf2)
    else:
        return get_scale_factor_l2(restraint, certainty, safety_factor, criteria, metal_loss_location, weld_interaction_sf, tables_path)

def get_restraint(quad_values: list, od: float, dent_depth_pct: float) -> str:
    """
    Determine the Restraint string based on the maximum RP value and Shallow/Deep condition
    
    Args:
        quad_values: List of quadrant values (replaces Excel Range)
        od: Outer diameter
        dent_depth_pct: Dent depth percentage
    
    Returns:
        Restraint condition string or "Error"
    """
    try:
        # Input validation
        if not quad_values or not all([od, dent_depth_pct]):
            return "Error"
        
        # Filter out NaN values when calculating max
        valid_values = []
        for value in quad_values:
            if pd.notna(value) and isinstance(value, (int, float)):
                valid_values.append(value)
        
        if not valid_values:
            return "Error"
        
        max_value = max(valid_values)
        
        # Determine if shallow or deep based on OD
        if od <= 12.75:
            is_shallow = dent_depth_pct < 4
        else:  # od > 12.75
            is_shallow = dent_depth_pct < 2.5
        
        # Additional validation
        if not isinstance(max_value, (int, float)) or max_value == 0:
            return "Error"
        
        # Determine restraint condition based on max_value and depth
        if max_value < 15:
            return "Unrestrained"
        elif max_value > 25:
            if is_shallow:
                return "Shallow Restrained"
            else:
                return "Deep Restrained"
        elif 15 <= max_value <= 25:
            if is_shallow:
                return "Shallow Mixed"
            else:
                return "Deep Mixed"
        else:
            return "Error"
            
    except Exception:
        return "Error"
    
def get_criteria_type(metal_loss_interaction: bool, weld_interaction: bool) -> str:
    """
    Determine the criteria type based on metal loss and weld interaction flags
    
    Args:
        metal_loss_interaction: Boolean indicating if there's metal loss interaction
        weld_interaction: Boolean indicating if there's weld interaction
    
    Returns:
        Criteria type string: "Metal Loss", "Weld", "Plain", or "Multiple"
    """
    if metal_loss_interaction and not weld_interaction:
        return "Metal Loss"
    elif not metal_loss_interaction and weld_interaction:
        return "Weld"
    elif not metal_loss_interaction and not weld_interaction:
        return "Plain"
    else:  # Both are True
        return "Multiple"
    
def get_RL(damage: float, service_years: float, scale_factor: float, weld_sf: float, ml_rf: float) -> tuple[float, float]:
    """
    Calculate the Remaining Life (RL) and RL with Factors
    """
    RL = RL_sf = None
    if isinstance(scale_factor, (int, float)):
        RL = (1.0 - damage) / (damage / service_years)
        RL_sf = RL / (scale_factor * weld_sf * ml_rf)

    if not RL or not RL_sf:
        return (math.nan, math.nan)
    
    return (RL, RL_sf)

def get_total_damage(dd: rfa.DentData, km_values: list, prange_list: list, MD49_cycles: list, curve_selection: dict, fatigue_curves_path: str = os.path.join(tables_folder, table_fatigue)) -> list[float]:
    """
    Calculate total damage for all fatigue curve options.
    Args:
        dd: DentData object containing pipe and feature characteristics
        km_values: List of K_M values for each bin
        prange_list: List of pressure ranges (as percentages of SMYS) for each bin
        MD49_cycles: List of cycle counts for each bin
        curve_selection: Dictionary containing fatigue category and curve selection. If using BS, also includes SD value.
        fatigue_curves_path: Path to the JSON file containing fatigue curve parameters
    Returns:
        Total damage as a float or error string
    """
    try:
        # Make sure length of km_values, prange_list, and MD49_cycles are the same
        if not (len(km_values) == len(prange_list) == len(MD49_cycles)):
            return "Error: Input lists must have the same length"
        
        # Load fatigue curve parameters from JSON
        with open(fatigue_curves_path, 'r') as f:
            fatigue_curves = json.load(f)
        
        press_range_list = [(val/100) * dd.SMYS * 2 * dd.WT / dd.OD for val in prange_list]  # Pressure range in psi
        damage_results = 0.0
        
        # ABS Damage Calculations
        if curve_selection["Category"] == "ABS":
            for curve in fatigue_curves[curve_selection["Category"]]:
                if curve["Curve"] == curve_selection["Curve"]:
                    curve_data = curve
                    break
            a_ksi = curve_data["A_ksi"]
            m = curve_data["m"]
            c_ksi = curve_data["C_ksi"]
            r = curve_data["r"]
            seq_ksi = curve_data["Seq_ksi"]

            for i in range(len(km_values)):
                # Peak Stress Range (ksi)
                s_ksi = km_values[i] * (press_range_list[i] * dd.OD / (2 * dd.WT)) / 1000  # Convert psi to ksi (1 ksi = 1000 psi)
                
                if s_ksi < seq_ksi:
                    cycles_to_failure = c_ksi * (s_ksi ** (-r))
                else:
                    cycles_to_failure = a_ksi * (s_ksi ** (-m))
                
                if cycles_to_failure > 0:
                    damage_results += MD49_cycles[i] / cycles_to_failure
        
        # DNV Damage Calculations
        if curve_selection["Category"] == "DNV":
            for curve in fatigue_curves[curve_selection["Category"]]:
                if curve["Curve"] == curve_selection["Curve"]:
                    curve_data = curve
                    break
            m1 = curve_data["m1"]
            loga1 = curve_data["loga1"]
            m2 = curve_data["m2"]
            loga2 = curve_data["loga2"]
            seq_mpa = curve_data["Seq"]

            for i in range(len(km_values)):
                # Peak Stress Range (ksi)
                s_ksi = km_values[i] * (press_range_list[i] * dd.OD / (2 * dd.WT)) / 1000  # Convert psi to ksi (1 ksi = 1000 psi)

                # Convert to MPa for DNV and BS calculations
                s_mpa = s_ksi * 6.89476  # Convert from ksi to MPa (1 ksi = 6.89476 MPa)
                
                if s_mpa > seq_mpa:
                    exp_a = loga1 - m1 * math.log10(s_mpa)
                else:
                    exp_a = loga2 - m2 * math.log10(s_mpa)
                
                cycles_to_failure = 10 ** exp_a
                
                if cycles_to_failure > 0:
                    damage_results += MD49_cycles[i] / cycles_to_failure
        
        # BS Damage Calculations
        if curve_selection["Category"] == "BS":
            for curve in fatigue_curves[curve_selection["Category"]]:
                if curve["Curve"] == curve_selection["Curve"]:
                    curve_data = curve
                    break
            log10c0 = curve_data["log10C0"]
            m = curve_data["m"]
            sd = curve_data["SD"]
            soc = curve_data["Soc"]
            
            # Use d = 2 as standard deviation factor (common practice)
            d = curve_selection["SD"] if "SD" in curve_selection else 2

            for i in range(len(km_values)):
                # Peak Stress Range (ksi)
                s_ksi = km_values[i] * (press_range_list[i] * dd.OD / (2 * dd.WT)) / 1000  # Convert psi to ksi (1 ksi = 1000 psi)

                # Convert to MPa for DNV and BS calculations
                s_mpa = s_ksi * 6.89476  # Convert from ksi to MPa (1 ksi = 6.89476 MPa)
            
                if s_mpa > soc:
                    cycles_to_failure = 10 ** (log10c0 - d * sd - m * math.log10(s_mpa))
                    
                if cycles_to_failure > 0:
                    damage_results += MD49_cycles[i] / cycles_to_failure

        return damage_results
    except Exception:
        traceback.print_exc()
        return "Error in calculation"


def process(dd: rfa.DentData, pf: md49.CreateProfiles, rainflow_results: tuple, curve_selection: dict, save_to_excel: bool = False, excel_path: str = "") -> dict | str:
    """
    Main processing function to read pipe characteristics, rainflow data, and feature sizing from MD-4-9 profiles.
    Computes and outputs the damage results. 
    Optional: Save a copy of the results to the Excel RLA document.
    Args:
        dd: DentData object containing pipe and feature characteristics
        pf: CreateProfiles object containing MD-4-9 profile results
        rainflow_results: Tuple containing rainflow cycle data, such as SSI, CI, MD49_SSI, cycles, and MD49_bins
        curve_selection: Dictionary containing curve selection parameters
        save_to_excel: Boolean flag to save results to Excel
        excel_path: Path to the Excel file to save results
    """

    try:
        SSI, CI, MD49_SSI, cycles, MD49_bins = rainflow_results
        # Perform calculations and populate results
        min_pressure_mean = min(cycles[:, 1])
        max_pressure_mean = max(cycles[:, 1])
        pressure_mean = np.mean(cycles[:, 1])
        max_pressure_range = max(cycles[:, 0])
        ili_pressure_psmys = 100* (dd.ili_pressure * dd.OD / (2 * dd.WT)) / dd.SMYS
        pmean_list_short = [25, 45, 65]
        closest_pmean = min(pmean_list_short, key=lambda x: abs(x - pressure_mean))

        rp_US_CCW = md49.get_restraint_parameter(pf._results_axial_us["areas"][15],
                                                 pf._results_circ_us_ccw["areas"][15],
                                                 pf._results_circ_us_ccw["lengths"][70]["length"],
                                                 pf._results_axial_us["lengths"][15]["length"],
                                                 pf._results_axial_us["lengths"][30]["length"],
                                                 pf._results_axial_us["lengths"][50]["length"],
                                                 pf._results_circ_us_ccw["lengths"][80]["length"])
        rp_US_CW = md49.get_restraint_parameter(pf._results_axial_us["areas"][15],
                                                 pf._results_circ_us_cw["areas"][15],
                                                 pf._results_circ_us_cw["lengths"][70]["length"],
                                                 pf._results_axial_us["lengths"][15]["length"],
                                                 pf._results_axial_us["lengths"][30]["length"],
                                                 pf._results_axial_us["lengths"][50]["length"],
                                                 pf._results_circ_us_cw["lengths"][80]["length"])
        rp_DS_CCW = md49.get_restraint_parameter(pf._results_axial_ds["areas"][15],
                                                 pf._results_circ_ds_ccw["areas"][15],
                                                 pf._results_circ_ds_ccw["lengths"][70]["length"],
                                                 pf._results_axial_ds["lengths"][15]["length"],
                                                 pf._results_axial_ds["lengths"][30]["length"],
                                                 pf._results_axial_ds["lengths"][50]["length"],
                                                 pf._results_circ_ds_ccw["lengths"][80]["length"])
        rp_DS_CW = md49.get_restraint_parameter(pf._results_axial_ds["areas"][15],
                                                 pf._results_circ_ds_cw["areas"][15],
                                                 pf._results_circ_ds_cw["lengths"][70]["length"],
                                                 pf._results_axial_ds["lengths"][15]["length"],
                                                 pf._results_axial_ds["lengths"][30]["length"],
                                                 pf._results_axial_ds["lengths"][50]["length"],
                                                 pf._results_circ_ds_cw["lengths"][80]["length"])
        calc_restraint = get_restraint([rp_US_CCW, rp_US_CW, rp_DS_CCW, rp_DS_CW], dd.OD, dd.dent_depth_percent)
        
        # MD-4-9 Prange list
        pmin_list = [10,10,10,10,10,10,10,20,20,20,20,20,20,30,30,30,30,30,40,40,40,40,50,50,50,60,60,70]
        pmax_list = [20,30,40,50,60,70,80,30,40,50,60,70,80,40,50,60,70,80,50,60,70,80,60,70,80,70,80,80]
        prange_list = [10,20,30,40,50,60,70,10,20,30,40,50,60,10,20,30,40,50,10,20,30,40,10,20,30,10,20,10]
        pmean_list = [15,20,25,30,35,40,45,25,30,35,40,45,50,35,40,45,50,55,45,50,55,60,55,60,65,65,70,75]

        # Create lists of results
        l0_Km_result = get_km_l0(dd.OD, dd.WT, calc_restraint, max_pressure_mean)
        l05_Km_results = [0] * len(prange_list)
        for i, val in enumerate(prange_list):
            l05_Km_results[i] = get_km_l05(calc_restraint, dd.OD, dd.WT, val)
        l05p_Km_results = [0] * len(prange_list)
        for i, _ in enumerate(prange_list):
            l05p_Km_results[i] = get_km_l05p(dd.OD, dd.WT, prange_list[i], pmean_list[i], calc_restraint)
        l075_Km_results = [0] * len(prange_list)
        for i, val in enumerate(prange_list):
            l075_Km_results[i] = get_km_l075(dd.OD, dd.WT, val, dd.dent_depth_percent, calc_restraint)
        l075p_Km_results = [0] * len(prange_list)
        for i, _ in enumerate(prange_list):
            l075p_Km_results[i] = get_km_l075p(dd.OD, dd.WT, prange_list[i], pmean_list[i], dd.dent_depth_percent, calc_restraint)
        l1_N_result = get_n_l1(calc_restraint, dd.OD, dd.WT, closest_pmean, ili_pressure_psmys, dd.SMYS/1000, 
                               pf.US_LAX, pf.US_AAX, pf.DS_LAX, pf.DS_AAX, 
                               pf.US_CCW_LTR, pf.US_CCW_ATR, pf.US_CW_LTR, pf.US_CW_ATR,
                               pf.DS_CCW_LTR, pf.DS_CCW_ATR, pf.DS_CW_LTR, pf.DS_CW_ATR)
        l2_N_results = [0] * len(prange_list)
        for i, _ in enumerate(pmin_list):
            l2_N_results[i] = get_n_l2(calc_restraint, dd.OD, dd.WT, pmean_list[i], pmax_list[i], pmin_list[i], ili_pressure_psmys, dd.SMYS/1000,
                                       pf.US_LAX, pf.US_AAX, pf.DS_LAX, pf.DS_AAX, 
                                       pf.US_CCW_LTR, pf.US_CCW_ATR, pf.US_CW_LTR, pf.US_CW_ATR,
                                       pf.DS_CCW_LTR, pf.DS_CCW_ATR, pf.DS_CW_LTR, pf.DS_CW_ATR)
        l2_Km_results = [0] * len(prange_list)
        lax30_us = pf._results_axial_us["lengths"][30]["length"]
        lax75_us = pf._results_axial_us["lengths"][75]["length"]
        lax85_us = pf._results_axial_us["lengths"][85]["length"]
        aax30_us = pf._results_axial_us["areas"][30]
        aax75_us = pf._results_axial_us["areas"][75]
        aax85_us = pf._results_axial_us["areas"][85]
        lax30_ds = pf._results_axial_ds["lengths"][30]["length"]
        lax75_ds = pf._results_axial_ds["lengths"][75]["length"]
        lax85_ds = pf._results_axial_ds["lengths"][85]["length"]
        aax30_ds = pf._results_axial_ds["areas"][30]
        aax75_ds = pf._results_axial_ds["areas"][75]
        aax85_ds = pf._results_axial_ds["areas"][85]
        ltr75_us_cw = pf._results_circ_us_cw["lengths"][75]["length"]
        atr75_us_cw = pf._results_circ_us_cw["areas"][75]
        ltr85_us_cw = pf._results_circ_us_cw["lengths"][85]["length"]
        ltr75_us_ccw = pf._results_circ_us_ccw["lengths"][75]["length"]
        atr75_us_ccw = pf._results_circ_us_ccw["areas"][75]
        ltr85_us_ccw = pf._results_circ_us_ccw["lengths"][85]["length"]
        ltr75_ds_cw = pf._results_circ_ds_cw["lengths"][75]["length"]
        atr75_ds_cw = pf._results_circ_ds_cw["areas"][75]
        ltr85_ds_cw = pf._results_circ_ds_cw["lengths"][85]["length"]
        ltr75_ds_ccw = pf._results_circ_ds_ccw["lengths"][75]["length"]
        atr75_ds_ccw = pf._results_circ_ds_ccw["areas"][75]
        ltr85_ds_ccw = pf._results_circ_ds_ccw["lengths"][85]["length"]
        for i, _ in enumerate(prange_list):
            l2_Km_results[i] = get_km_l2(dd.OD, dd.WT, calc_restraint,
                                         lax30_us, lax75_us, lax85_us, aax30_us, aax75_us, aax85_us, 
                                         lax30_ds, lax75_ds, lax85_ds, aax30_ds, aax75_ds, aax85_ds, 
                                         ltr75_us_cw, atr75_us_cw, ltr85_us_cw, ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                                         ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw, ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw,
                                         pmean_list[i], ili_pressure_psmys, prange_list[i]/100, pmean_list[i]/100, pmax_list[i]/100)
        
        km_params = (lax30_us, lax75_us, lax85_us,
                    aax30_us, aax75_us, aax85_us,
                    lax30_ds, lax75_ds, lax85_ds,
                    aax30_ds, aax75_ds, aax85_ds,
                    ltr75_us_cw, atr75_us_cw, ltr85_us_cw,
                    ltr75_us_ccw, atr75_us_ccw, ltr85_us_ccw,
                    ltr75_ds_cw, atr75_ds_cw, ltr85_ds_cw,
                    ltr75_ds_ccw, atr75_ds_ccw, ltr85_ds_ccw)
        l2_md24_unbinned_results = get_md24_unbinned(dd.OD, dd.WT, dd.SMYS, calc_restraint,
                                                     cycles, ili_pressure_psmys, km_params)
        # Calculate scale and safety factors
        cps_sf = 4.0 if dd.CPS else 2.0
        weld_sf = 10.0 if dd.interaction_weld == True else 5.0
        if not pd.isna(dd.ml_depth_percent):
            ml_rf = dd.WT/(dd.WT - (dd.ml_depth_percent/100)*dd.WT)
        else:
            ml_rf = 1.0
        criteria = get_criteria_type(dd.interaction_corrosion, dd.interaction_weld)
        l0_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "0")
        l05_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "0.5")
        l05p_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "0.5+")
        l075_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "0.75")
        l075p_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "0.75+")
        l1_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "1")
        l2_sf = get_scale_factor_md53(dd.confidence, cps_sf, criteria, "2")
        l2_md24_sf = get_scale_factor_l2_wrap(calc_restraint, dd.confidence, cps_sf, criteria, dd.ml_location, weld_sf)
        # Calculate damage/year, and then calculate remaining life. Apply all factors to determine remaining life with factors
        RLA_l0, RLA_l0_sf = get_RL(get_total_damage(dd, [l0_Km_result] * len(prange_list), prange_list, MD49_bins, curve_selection), dd.service_years, l0_sf, weld_sf, ml_rf)
        RLA_l05, RLA_l05_sf = get_RL(get_total_damage(dd, l05_Km_results, prange_list, MD49_bins, curve_selection), dd.service_years, l05_sf, weld_sf, ml_rf)
        RLA_l05p, RLA_l05p_sf = get_RL(get_total_damage(dd, l05p_Km_results, prange_list, MD49_bins, curve_selection), dd.service_years, l05p_sf, weld_sf, ml_rf)
        RLA_l075, RLA_l075_sf = get_RL(get_total_damage(dd, l075_Km_results, prange_list, MD49_bins, curve_selection), dd.service_years, l075_sf, weld_sf, ml_rf)
        RLA_l075p, RLA_l075p_sf = get_RL(get_total_damage(dd, l075p_Km_results, prange_list, MD49_bins, curve_selection), dd.service_years, l075p_sf, weld_sf, ml_rf)
        if closest_pmean == 25 or closest_pmean == 45:
            damage_val = (SSI / l1_N_result) * (13000/(30 * dd.SMYS / 100)) ** 3
        else:
            damage_val = (SSI / l1_N_result) * (13000/(40 * dd.SMYS / 100)) ** 3
        RLA_l1, RLA_l1_sf = get_RL(damage_val, dd.service_years, l1_sf, weld_sf, ml_rf)
        damage_val = [MD49_bins[i] / l2_N_results[i] for i in range(len(l2_N_results))]
        damage_val = sum(damage_val)
        RLA_l2, RLA_l2_sf = get_RL(damage_val, dd.service_years, l2_sf, weld_sf, ml_rf)
        RLA_l2_md24, RLA_l2_md24_sf = get_RL(get_total_damage(dd, l2_Km_results, prange_list, MD49_bins, curve_selection), dd.service_years, l2_md24_sf, weld_sf, ml_rf)
        RLA_l2_md24_unbinned = {
                                    "ABS": {},
                                    "DNV": {},
                                    "BS": {}
                                }
        RLA_l2_md24_unbinned_sf = {
                                    "ABS": {},
                                    "DNV": {},
                                    "BS": {}
                                }
        # Determine the Remaining Life (RL) for each fatigue curve category and curve option
        for category in ["ABS", "DNV", "BS"]:
            for curve, damage in l2_md24_unbinned_results[category].items():
                RLA_l2_md24_unbinned[category][curve], RLA_l2_md24_unbinned_sf[category][curve] = get_RL(float(damage), dd.service_years, l2_md24_sf, weld_sf, ml_rf)
        # Export results to Excel if specified
        if save_to_excel and excel_path:
            wb = openpyxl.load_workbook(filename=rla_template, read_only=False)

            # Select the Summary sheet
            wbs = wb['Summary']
            # Write pipe characteristics to Summary sheet
            wbs['D4'] = round(dd.OD, 3)
            wbs['D5'] = round(dd.WT, 3)
            wbs['D6'] = round(dd.SMYS, -3) # Round to nearest 1000 so we get 70,000
            wbs['D8'] = round(dd.MAOP, 0)
            wbs['D9'] = round(dd.service_years, 3)
            wbs['D10'] = float(dd.min_range)
            wbs['H4'] = str(dd.dent_category)
            wbs['H5'] = str(dd.dent_ID)
            
            # Select the Rainflow sheet
            wbs = wb['Rainflow']
            # Save rainflow summary data to the sheet
            wbs['K7'] = float(min_pressure_mean)
            wbs['K8'] = float(max_pressure_mean)
            wbs['K10'] = float(max_pressure_range)
            wbs['K21'] = float(SSI)
            wbs['K24'] = float(CI)
            wbs['K27'] = float(MD49_SSI)
            # Save pipe tally information to the sheet
            wbs['K61'] = float(dd.ili_pressure)
            wbs['K63'] = float(dd.dent_depth_percent)
            wbs['K65'] = float(pressure_mean)
            wbs['K67'] = str(dd.interaction_corrosion)
            wbs['K68'] = float(dd.ml_depth_percent)
            wbs['K69'] = str(dd.ml_location)
            wbs['K70'] = str(dd.interaction_weld)
            wbs['K72'] = float(dd.confidence)
            wbs['K73'] = str(dd.CPS)
            wbs['K83'] = str(calc_restraint)
            wbs['K84'] = str(dd.restraint_condition)

            # Row range for pasting results
            start_row = 3
            end_row = 30
            # Column letters for pasting results
            MD49_bins_col = "AK"
            l0_col = "AR"
            l05_col = "AZ"
            l05p_col = "BH"
            l075_col = "BP"
            l075p_col = "BX"
            l2_col = "CH"
            l2_md24_col = "CJ"
            # Bin the cycles from rainflow_results to 60 bins, and filter out anything below the min_range
            # Create 60 bins spanning the min_range to max_range
            # Paste the results to the appropriate columns
            bins_60 = np.linspace(dd.min_range, max_pressure_range, 61)
            filtered_cycles = cycles[cycles[:, 0] >= dd.min_range]
            cycles_60, _ = np.histogram(filtered_cycles[:, 0], bins=bins_60)
            cells_range = [f"M{row}" for row in range(start_row, end_row + 1)]
            cells_cycles = [f"N{row}" for row in range(start_row, end_row + 1)]
            for c_r, bin_val, c_c, cycle_val in zip(cells_range, bins_60[:-1], cells_cycles, cycles_60):
                wbs[c_r] = bin_val
                wbs[c_c] = cycle_val

            # Save the MD49_bins to the sheet
            cells = [f"{MD49_bins_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, bin_val in zip(cells, MD49_bins):
                wbs[cell] = bin_val
            # Level 0
            cells = [f"{l0_col}{row}" for row in range(start_row, end_row + 1)]
            for cell in cells:
                wbs[cell] = l0_Km_result
            # Level 0.5
            cells = [f"{l05_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, km_val in zip(cells, l05_Km_results):
                wbs[cell] = km_val
            # Level 0.5+
            cells = [f"{l05p_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, km_val in zip(cells, l05p_Km_results):
                wbs[cell] = km_val
            # Level 0.75
            cells = [f"{l075_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, km_val in zip(cells, l075_Km_results):
                wbs[cell] = km_val
            # Level 0.75+
            cells = [f"{l075p_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, km_val in zip(cells, l075p_Km_results):
                wbs[cell] = km_val
            # Level 1
            wbs["CF3"] = l1_N_result
            # Level 2
            cells = [f"{l2_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, N_val in zip(cells, l2_N_results):
                wbs[cell] = N_val
            # Level 2 (MD-2-4)
            cells = [f"{l2_md24_col}{row}" for row in range(start_row, end_row + 1)]
            for cell, km_val in zip(cells, l2_Km_results):
                wbs[cell] = km_val

            # Save the profile lengths and areas to the sheet
            LAX_US_start = "AH33"
            AAX_US_start = "AJ33"
            LAX_DS_start = "AM33"
            AAX_DS_start = "AO33"
            LTR_US_CCW_start = "AH47"
            ATR_US_CCW_start = "AJ47"
            LTR_US_CW_start = "AM47"
            ATR_US_CW_start = "AO47"
            LTR_DS_CCW_start = "AH61"
            ATR_DS_CCW_start = "AJ61"
            LTR_DS_CW_start = "AM61"
            ATR_DS_CW_start = "AO61"
            profile_cell_lists = [LAX_US_start, AAX_US_start, LAX_DS_start, AAX_DS_start,
                                  LTR_US_CCW_start, ATR_US_CCW_start, LTR_US_CW_start, ATR_US_CW_start,
                                  LTR_DS_CCW_start, ATR_DS_CCW_start, LTR_DS_CW_start, ATR_DS_CW_start]
            profile_val_lists = [pf.US_LAX, pf.US_AAX, pf.DS_LAX, pf.DS_AAX,
                                 pf.US_CCW_LTR, pf.US_CCW_ATR, pf.US_CW_LTR, pf.US_CW_ATR,
                                 pf.DS_CCW_LTR, pf.DS_CCW_ATR, pf.DS_CW_LTR, pf.DS_CW_ATR]
            for cell_start, val_list in zip(profile_cell_lists, profile_val_lists):
                col = ''.join(filter(str.isalpha, cell_start))
                row_start = int(''.join(filter(str.isdigit, cell_start)))
                cells = [f"{col}{row}" for row in range(row_start, row_start + len(val_list))]
                for cell, val in zip(cells, val_list):
                    wbs[cell] = val

            # Save the Scale Factors for each Level
            wbs['K88'] = l0_sf
            wbs['K93'] = l05_sf
            wbs['K98'] = l05p_sf
            wbs['K103'] = l075_sf
            wbs['K108'] = l075p_sf
            wbs['K113'] = l1_sf
            wbs['K118'] = l2_sf
            wbs['K123'] = l2_md24_sf
            wbs['K131'] = l2_md24_sf

            # Level 2 (MD-2-4 Unbinned). Paste direct total damage results to the cells
            wbs['K125'] = l2_md24_unbinned_results["ABS"]["D"]
            wbs['K126'] = l2_md24_unbinned_results["DNV"]["C"]
            wbs['K127'] = l2_md24_unbinned_results["BS"]["D"]

            # Save as a copy
            wb.save(excel_path)

        # Return results as a dictionary
        results = {
            "Calculated Restraint": calc_restraint,
            "Quadrant RP Values": {
                "US-CCW": rp_US_CCW,
                "US-CW": rp_US_CW,
                "DS-CCW": rp_DS_CCW,
                "DS-CW": rp_DS_CW
            },
            "Life": {
                "0": {"No SF": RLA_l0, "Yes SF": RLA_l0_sf},
                "0.5": {"No SF": RLA_l05, "Yes SF": RLA_l05_sf},
                "0.5+": {"No SF": RLA_l05p, "Yes SF": RLA_l05p_sf},
                "0.75": {"No SF": RLA_l075, "Yes SF": RLA_l075_sf},
                "0.75+": {"No SF": RLA_l075p, "Yes SF": RLA_l075p_sf},
                "1": {"No SF": RLA_l1, "Yes SF": RLA_l1_sf},
                "2": {"No SF": RLA_l2, "Yes SF": RLA_l2_sf},
                "2_md24": {"No SF": RLA_l2_md24, "Yes SF": RLA_l2_md24_sf},
                "2_md24_unbinned": {"No SF": RLA_l2_md24_unbinned, "Yes SF": RLA_l2_md24_unbinned_sf}
            },
        }
        return results

    except Exception as e:
        traceback.print_exc()
        return f"Error in processing: {e}"


