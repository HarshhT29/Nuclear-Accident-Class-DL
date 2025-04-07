# Nuclear Power Plant Accident Classification

## Project Overview

This project develops a deep learning-based accident classifier for nuclear power plants using the NPPAD dataset. The system analyzes time-series data of operational parameters to identify the type of accident occurring in the plant.

### Dataset

The NPPAD (Nuclear Power Plant Accident Dataset) contains simulations of 12 accident types:
- LOCA (Loss of Coolant Accident)
- SGTR (Steam Generator Tube Rupture)
- MSLB (Main Steam Line Break)
- LOFW (Loss of Feedwater)
- LOOP (Loss of Offsite Power)
- TMLB (Station Blackout)
- SLOCA (Small Loss of Coolant Accident)
- SLBO (Steam Line Break Outside)
- RCP (Reactor Coolant Pump seizure)
- LODCP (Loss of DC Power)
- IISLOCA (Interfacing System LOCA)
- LOHT (Loss of Heat Transfer)

Each accident type has 100 simulations of varying severity levels. Each simulation consists of:
1. A CSV file with ~97 operational parameters recorded at 10-second intervals
2. A TransientReport.txt file describing subsystem actions during the simulation

## One-Day Implementation Plan

This repository contains a condensed implementation of an LSTM/GRU model that can be completed in a single day. The implementation follows this step-by-step approach:

### Morning (First 4 Hours)
1. Data Setup & Exploration
2. Data Preprocessing

### Afternoon (Next 4 Hours)
3. Model Implementation
4. Training & Validation
5. Refinement & Evaluation
6. Documentation

## Project Structure

```
.
├── README.md                          # Project overview and setup instructions
├── scripts/                           # Python implementation scripts
│   ├── data_exploration.py            # Script for exploring the NPPAD dataset
│   └── data_preprocessing.py          # Script for preprocessing the data
├── docs/                              # Documentation files
│   ├── data_exploration_explanation.md # Detailed explanation of exploration script
│   └── data_preprocessing_explanation.md # Detailed explanation of preprocessing script
├── exploration_results/               # Output directory for exploration results
│   ├── parameter_statistics.csv       # Statistics for all parameters
│   ├── top_parameters.txt             # Ranked list of most distinctive parameters
│   └── *.png                          # Visualizations of key parameters
├── processed_data/                    # Preprocessed data for model training
│   ├── X_train.npy                    # Training features
│   ├── y_train.npy                    # Training labels
│   ├── X_val.npy                      # Validation features
│   ├── y_val.npy                      # Validation labels
│   ├── X_test.npy                     # Test features
│   ├── y_test.npy                     # Test labels
│   ├── meta_*.json                    # Metadata for each split
│   ├── scaler.pkl                     # Fitted scaler for normalization
│   └── preprocessing_info.json        # Documentation of preprocessing settings
└── NPPAD/                             # The original dataset (not included in repo)
    ├── LOCA/                          # Accident type folders
    │   ├── 1.csv                      # Simulation data
    │   ├── 1TransientReport.txt       # Simulation report
    │   └── ...
    ├── SGTR/
    └── ...
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/nuclear-accident-classification.git
   cd nuclear-accident-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the NPPAD dataset:
   - Place the NPPAD dataset in the root directory, maintaining its structure
   - Ensure each accident type folder contains its respective simulation files

## Usage

### 1. Data Exploration (Hour 1)

Run the data exploration script to analyze the dataset and identify key parameters:

```bash
python scripts/data_exploration.py
```

This will:
- Verify dataset integrity
- Calculate statistics for all parameters
- Identify the most distinctive parameters
- Create visualizations of key patterns
- Save results to the `exploration_results` directory

### 2. Data Preprocessing (Hour 2)

Preprocess the data to prepare it for model training:

```bash
python scripts/data_preprocessing.py
```

This will:
- Create standardized, windowed data from the raw time-series
- Split data into training, validation, and test sets
- Apply normalization
- Save processed data to the `processed_data` directory

### 3. Model Implementation (Future Hours)

The subsequent scripts for model implementation, training, and evaluation will be developed in Hours 3-8 of the one-day plan.

## Project Goals

1. Develop a working prototype of an LSTM/GRU-based classifier
2. Achieve classification accuracy better than random guessing
3. Understand model performance across different accident types
4. Identify areas for future improvement

## Future Improvements

For a more comprehensive implementation beyond the one-day scope:
- Implement more complex architectures (transformers, CNN-LSTM hybrids)
- Add attention mechanisms for better interpretability
- Conduct more extensive hyperparameter tuning
- Develop a deployment pipeline for real-time classification

## License

This project is provided for educational purposes only. The NPPAD dataset should be used according to its original licensing terms.

## Acknowledgments

- The NPPAD dataset providers for enabling nuclear accident research
- The nuclear safety community for advancing accident detection and prevention techniques

## Data Preprocessing

The preprocessed data files are not included in this repository due to size limitations. To generate these files:

1. Run the data exploration script:
   ```bash
   python scripts/data_exploration.py
   ```

2. Run the data preprocessing script:
   ```bash
   python scripts/data_preprocessing.py
   ```

This will create the following files in the `processed_data/` directory:
- `X_train.npy`, `y_train.npy` - Training data
- `X_val.npy`, `y_val.npy` - Validation data
- `X_test.npy`, `y_test.npy` - Test data
- `scaler.pkl` - Fitted StandardScaler object
- `preprocessing_info.json` - Preprocessing configuration and statistics 