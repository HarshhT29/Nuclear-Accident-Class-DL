# Data Preprocessing Script Explanation

## Overview

`data_preprocessing.py` transforms the raw NPPAD dataset into a format suitable for training LSTM/GRU models. It handles data loading, feature selection, windowing, normalization, and train-validation-test splitting, creating a ready-to-use dataset for model training.

## Functions and Workflow

### 1. `create_output_directory()`

Creates a directory to store the processed data.

**Purpose:**
- Ensures a dedicated location for all processed data files
- Prevents potential file path errors during saving

### 2. `load_top_parameters()`

Loads the list of top parameters identified during the exploration phase.

**Purpose:**
- Reads the parameter rankings from the exploration output
- Extracts parameter names from the ranked list
- Enables feature selection based on exploratory analysis
- Falls back to using all parameters if the file is not found

### 3. `load_accident_data()`

Loads and performs initial preprocessing for a specific accident type.

**Purpose:**
- Loads all CSV files for a given accident type
- Filters to include only the top parameters (if specified)
- Handles missing values using forward/backward fill and zero replacement
- Tracks file IDs for metadata

### 4. `create_windows()`

Segments time-series data into fixed-length windows with overlap.

**Purpose:**
- Creates windowed samples from continuous time-series data
- Applies a fixed window size (default: 30 time steps)
- Implements overlap between consecutive windows (default: 10 time steps)
- Handles shorter samples through padding or skipping

### 5. `preprocess_all_data()`

Orchestrates preprocessing for all accident types.

**Purpose:**
- Processes each accident type in sequence
- Creates windowed data for all samples
- Assigns labels based on accident type
- Generates metadata for traceability
- Consolidates all data into combined arrays

### 6. `normalize_data()`

Normalizes the data using standardization (z-score normalization).

**Purpose:**
- Fits a StandardScaler to the training data only
- Applies the same scaling to validation and test data
- Handles the 3D shape of the windowed time-series data
- Returns the scaler for later use during inference

### 7. `split_data()`

Splits the dataset into training, validation, and test sets.

**Purpose:**
- Creates a 70/15/15 split for train/val/test by default
- Ensures stratification to maintain class balance
- Splits corresponding metadata alongside the data
- Reports the size of each resulting dataset

### 8. `save_processed_data()`

Saves all processed data and related information to files.

**Purpose:**
- Saves numpy arrays for features and labels
- Stores metadata in JSON format
- Saves the fitted scaler for future use
- Creates a preprocessing info file documenting all settings

### 9. `main()`

Coordinates the entire preprocessing pipeline.

**Purpose:**
- Executes each preprocessing step in sequence
- Reports progress and key statistics
- Returns dataset information for model development

## Key Features

1. **Feature Selection**: Uses only the most informative parameters identified during exploration
2. **Windowing**: Creates fixed-length segments with overlap for sequence modeling
3. **Missing Value Handling**: Applies forward/backward fill and zero replacement
4. **Standardization**: Normalizes features to zero mean and unit variance
5. **Stratified Splitting**: Maintains class balance across train/val/test sets
6. **Metadata Tracking**: Preserves information about the source of each window

## Outputs

The script creates a `processed_data` directory containing:

- **X_train.npy, X_val.npy, X_test.npy**: Feature arrays for each split
- **y_train.npy, y_val.npy, y_test.npy**: Label arrays for each split
- **meta_train.json, meta_val.json, meta_test.json**: Metadata for each split
- **scaler.pkl**: The fitted StandardScaler for normalization
- **preprocessing_info.json**: Documentation of preprocessing settings and statistics

## Usage

Run this script after data exploration to prepare the dataset for model training:

```bash
python data_preprocessing.py
```

The resulting preprocessed data is ready to be fed directly into an LSTM/GRU model. 