# -*- coding: utf-8 -*-
"""
Convert Excel sheets to JSON files.
"""

import pandas as pd
import json
import os

# Loop through all .xlsx documents in the directory that begin with "Tables_", load the Excel file, then convert each sheet to JSON
# Use the same directory as the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for file in os.listdir():
    if file.startswith("Tables_") and file.endswith(".xlsx"):
        xls = pd.ExcelFile(file)

        # Dictionary to hold all tables
        tables_dict = {}

        # Iterate over each sheet (table)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            # Convert DataFrame to list of dicts (handles varying rows/columns)
            tables_dict[sheet_name] = df.to_dict(orient='records')

        # Save as JSON using the original file name with .json extension
        json_filename = file.replace('.xlsx', '.json')
        # Save the JSON file to the same directory as the script
        # If the file already exists, overwrite it
        with open(json_filename, 'w') as f:
            json.dump(tables_dict, f, indent=4)