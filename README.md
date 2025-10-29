# Pipeline Dents MD-2-4 & MD-5-3 Processing Workflow

## Overview

This project provides comprehensive tools and scripts for processing pipeline pressure history data, performing rainflow analysis, smoothing ILI caliper data, and calculating fatigue damage using MD-2-4 and MD-5-3 methods. The workflow is designed to process large datasets of pipeline inspection data, pressure histories, and caliper measurements to generate detailed fatigue assessments and remaining life analyses.

---

## New Integrated Workflow (batch_v3.py)

### **Unified Processing Pipeline**
The new `batch_v3.py` script consolidates the entire workflow into a single, streamlined process that:

1. **Matches Dents to Stations**: Automatically finds upstream and downstream pressure monitoring stations for each dent based on continuous measure positions
2. **Extracts Pressure Histories**: Retrieves relevant pressure data columns for rainflow analysis
3. **Performs Rainflow Analysis**: Calculates stress cycles and fatigue damage using industry-standard methods
4. **Applies MD-2-4 Method**: Calculates stress concentration factors (Km) based on dent geometry and restraint conditions
5. **Applies MD-5-3 Method**: Determines scale factors and performs remaining life analysis
6. **Processes Caliper Data**: Smooths raw ILI measurements and extracts MD-4-9 geometric parameters
7. **Generates Comprehensive Reports**: Creates detailed Excel workbooks for each dent with all analysis results

---

## Input Requirements

### **Required Files**
1. **Pipe Tally File** (.xlsx): Contains dent information with columns:
   - `AP Measure (m)`: Absolute position of each dent
   - `Feature Type`: Type of pipeline feature
   - `OD (mm)`, `WT (mm)`, `SMYS (MPa)`: Pipe specifications
   - Additional dent geometry parameters

2. **Pump Stations File** (.xlsx): Contains pressure monitoring station data:
   - `Continuous Measure (m)`: Station positions
   - `Tag Name`: Station identifiers matching pressure history column headers

3. **Caliper Folder**: Directory containing ILI caliper data files (.xlsx)
   - Raw measurement data for dent geometry analysis
   - Files should be named to match dent identifiers

4. **Output Folder**: Directory where individual dent analysis workbooks will be saved

5. **Summary Folder**: Directory for summary reports and consolidated results

### **Optional Files**
- **Press Dict File** (.xlsx/.json): Custom pressure bin definitions for specialized analysis

---

## Processing Methods

### **MD-2-4 Fatigue Analysis**
- **Km Calculation**: Stress concentration factors based on:
  - Dent geometry (length, area, depth measurements)
  - Restraint conditions (unrestrained, shallow/deep restrained)
  - Pressure loading conditions
- **Fatigue Curves**: Support for ABS, DNV, and BS fatigue assessment standards
- **Damage Accumulation**: Miner's rule implementation for multiple stress levels

### **MD-5-3 Remaining Life Analysis**
- **Scale Factors**: Uncertainty and safety factor calculations
- **Confidence Levels**: Multiple certainty levels (50%, 80%, 90%)
- **Assessment Levels**: Level 0 through Level 2 analyses
- **Interaction Effects**: Metal loss and weld interaction considerations

### **MD-4-9 Geometric Analysis**
- **Data Smoothing**: Advanced filtering of raw caliper measurements
- **Length Calculations**: Dent extent determination at multiple depth thresholds
- **Area Calculations**: Cross-sectional area reductions
- **Quadrant Analysis**: Separate processing for upstream/downstream and CW/CCW orientations

---

## Output Files

### **Individual Dent Workbooks**
Each processed dent generates a comprehensive Excel workbook containing:

1. **Summary Sheet**: Key results and assessment outcomes
2. **Pressure History**: Extracted and processed pressure data
3. **Rainflow Results**: Cycle counting and stress range analysis
4. **MD-2-4 Analysis**: Km calculations and fatigue damage assessment
5. **MD-5-3 Analysis**: Scale factors and remaining life calculations
6. **Smoothed Data**: Processed caliper measurements
7. **MD-4-9 Results**: Length and area calculations by quadrant
8. **Charts and Visualizations**: Graphical representations of key results

### **Summary Reports**
- **Processing Log**: Detailed record of all operations performed
- **Batch Summary**: Consolidated results across all processed dents
- **Error Reports**: Documentation of any processing issues or failures

---

## Installation and Setup

### **Dependencies**
```bash
pip install pandas numpy openpyxl matplotlib rainflow scipy xlwings
```