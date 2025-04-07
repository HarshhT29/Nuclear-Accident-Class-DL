# Data Exploration Script Explanation

## Overview

`data_exploration.py` is designed to perform rapid initial exploration of the NPPAD nuclear accident dataset. It analyzes the structure, integrity, and key characteristics of the dataset to help identify the most informative operational parameters for the classification task.

## Functions and Workflow

### 1. `create_directory_structure()`

This function sets up the necessary directory structure and verifies that the dataset folders exist.

**Purpose:**
- Creates an output directory for exploration results
- Verifies that the main NPPAD data directory exists
- Checks if all 12 accident type folders are present
- Reports any missing folders

### 2. `check_data_integrity()`

Performs quick sampling checks to verify the completeness and readability of the data files.

**Purpose:**
- Counts CSV and TXT files for each accident type
- Verifies if the expected number of files (100 per accident type) exists
- Samples a CSV file from each accident type to check format
- Reports file counts and any reading errors

### 3. `load_sample_data()`

Loads a small subset of data (10 samples per accident type) for rapid exploration.

**Purpose:**
- Randomly selects a subset of files from each accident type
- Loads the selected files into dataframes
- Organizes data by accident type for further analysis
- Uses a small sample to enable quick processing

### 4. `calculate_basic_statistics()`

Calculates essential statistical measures for each parameter across accident types.

**Purpose:**
- Computes mean, standard deviation, min, and max for each parameter
- Organizes statistics by accident type to enable comparison
- Saves results to a CSV file for later reference
- Helps identify parameters with distinctive patterns

### 5. `identify_distinctive_parameters()`

Identifies parameters that show significant variance across different accident types.

**Purpose:**
- Calculates variance of mean values across accident types for each parameter
- Ranks parameters by variance (higher variance suggests better discriminatory power)
- Selects the top 20-30 most distinctive parameters
- Saves the ranked list to a text file for use in preprocessing

### 6. `create_visualizations()`

Creates visualizations to help understand parameter patterns across accident types.

**Purpose:**
- Generates time-series plots for the top 3 most distinctive parameters
- Creates a correlation heatmap to show relationships between parameters
- Helps identify visual patterns that distinguish accident types
- Saves plots to image files for documentation

### 7. `main()`

Orchestrates the entire exploration process in a logical sequence.

**Purpose:**
- Executes each step of the exploration process
- Reports progress and findings
- Returns the top parameters for use in preprocessing

## Key Features

1. **Fast Exploration**: Uses a subset of data to enable quick analysis
2. **Parameter Ranking**: Identifies the most informative parameters for classification
3. **Data Integrity Checking**: Verifies completeness and readability of dataset
4. **Visual Analysis**: Creates plots to visualize key parameter patterns
5. **Output Documentation**: Saves findings to files for future reference

## Outputs

- **parameter_statistics.csv**: Statistical measures for all parameters
- **top_parameters.txt**: Ranked list of most distinctive parameters
- **{param}_comparison.png**: Time-series plots for top parameters
- **correlation_heatmap.png**: Heatmap showing parameter correlations

## Usage

Run this script first before proceeding to data preprocessing to identify which parameters are most useful for the classification task.

```bash
python data_exploration.py
```

The script will create an `exploration_results` directory containing all output files. 