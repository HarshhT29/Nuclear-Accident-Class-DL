# Data Preprocessing Scripts

This directory contains scripts for data exploration and preprocessing of the nuclear accident dataset.

## Data Exploration (`data_exploration.py`)

This script analyzes the NPPAD dataset to identify informative parameters and patterns.

### Features:
- Dataset integrity verification
- Parameter statistics calculation
- Feature importance ranking
- Visualization generation

### Output:
- `exploration_results/parameter_statistics.csv`: Statistics for all parameters
- `exploration_results/top_parameters.txt`: Ranked list of most distinctive parameters
- `exploration_results/*.png`: Visualizations of key parameters

## Data Preprocessing (`data_preprocessing.py`)

This script transforms the raw data into a format suitable for model training.

### Features:
- Time window creation
- Data normalization
- Train/validation/test split
- Metadata generation

### Processing Steps:
1. Load raw data from CSV files
2. Create overlapping time windows
3. Handle missing values
4. Normalize features
5. Split data into train/validation/test sets
6. Save processed data and metadata

### Output Files:
- `processed_data/X_train.npy`: Training features
- `processed_data/y_train.npy`: Training labels
- `processed_data/X_val.npy`: Validation features
- `processed_data/y_val.npy`: Validation labels
- `processed_data/X_test.npy`: Test features
- `processed_data/y_test.npy`: Test labels
- `processed_data/meta_*.json`: Metadata for each split

### Parameters:
- Window size: 30 time steps
- Overlap: 15 time steps
- Train/Val/Test split: 70/15/15

## Usage

1. Run data exploration:
```bash
python data_exploration.py
```

2. Run data preprocessing:
```bash
python data_preprocessing.py
```

## Dependencies

- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.2
- Matplotlib >= 3.4.2
- Seaborn >= 0.11.1

## Data Format

### Input:
- CSV files containing time-series data
- Each file contains ~97 operational parameters
- Data recorded at 10-second intervals

### Output:
- Numpy arrays with shape (samples, window_size, features)
- Normalized values in range [-1, 1]
- Metadata containing preprocessing information 