# Pipeline Dents MD-4-9 Processing Workflow

## Overview

This project provides tools and scripts for processing pipeline pressure history data, performing rainflow analysis, smoothing ILI caliper data, and exporting MD-4-9 lengths and areas. The workflow is designed to help you filter, analyze, and export results from large datasets, particularly Excel files containing pipeline inspection and pressure data.

---

## Workflow Steps

### 1. **Input Data**
- Place your Excel files (e.g., pressure history, dent ILI caliper data) in the working directory into respective folders (e.g., Pressure History, Caliper Data).
- Ensure columns are named consistently (e.g., "Section", "Feature Type", "Tag Name", etc.).

### 2. **Pressure History Matching and Grouping**
- The `combine_histories.py` script combines all pressure histories belong to the same line segment.
- The script can group Excel files by matching column headers, allowing you to combine datasets from different years or sets.
- Closest column name matching is available for flexible data selection.
- For each dent, the workflow finds the nearest upstream and downstream stations based on distance, then extracts the corresponding pressure history columns for analysis.

### 3. **Rainflow Analysis**
- The `batch_rainflow.py` script performs rainflow counting and calculates fatigue metrics.
- An individual Excel document is saved for every dent feature based on the template .xlsm Excel document.
- This script depends on the custom `rainflow_analysis.py` module.
- Results are saved after each iteration for progress tracking.

### 4. **Raw ILI Caliper Data Smoothing**
- The `batch_smoothing.py` script performs smoothing of the raw ILI caliper data.
- A worksheet named `Smoothed Data` containing the results of the data smoothing will be saved to the individual Excel document.
- This script depends on the custom `processing.py` module.
- Results are saved after each iteration for progress tracking.

### 5. **MD-4-9 Lengths and Areas**
- The `batch_md49.py` script determines the corresponding lengths and areas of the dent feature based on the MD-4-9 method from [PR-214-114500-R01](https://www.prci.org/150177.aspx).
- Multiple worksheets containing the results from the four quadrants (US-CCW, US-CW, DS-CCW, DS-CW) will be saved to the individual Excel document.
- This script depends on the custom `API1183_v2.py` module.
- Results are saved after each iteration for progress tracking.

---

## Customization

- **Edit this README** to document your specific workflow, data sources, and any custom scripts you add.
- Adjust script parameters (e.g., column names, file paths) as needed for your datasets.

---

## Requirements

- Python 3.x
- pandas
- numpy
- openpyxl
- matplotlib
- rainflow
- scipy

Install dependencies with:
```
pip install pandas numpy openpyxl matplotlib rainflow scipy
```

---

## Notes

- Always check your column names and data types before running scripts.
- Save your results frequently to avoid data loss.
- For large datasets, monitor memory usage and consider processing in batches.

---

## To Do

- [ ] Add more detailed documentation for each script.
- [ ] Include sample data files.
- [ ] Expand error handling and logging.

---

*Edit this README to reflect your workflow and project