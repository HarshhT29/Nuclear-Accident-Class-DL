#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast data preprocessing script for nuclear accident classification
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
DATA_DIR = "NPPAD"  # Adjust if your dataset is located elsewhere
ACCIDENT_TYPES = ['LOCA', 'SGBTR', 'LR', 'MD', 'SGATR', 'SLBIC',
                     'LOCAC', 'RI', 'FLB', 'LLB', 'SLBOC','RW']   # All 12 accident types
PROCESSED_DIR = "processed_data"
WINDOW_SIZE = 30  # Using 30 time steps as fixed window size
OVERLAP = 10  # Overlap between consecutive windows
TOP_PARAMS_FILE = "exploration_results/top_parameters.txt"  # From exploration

def create_output_directory():
    """Create directory for processed data"""
    Path(PROCESSED_DIR).mkdir(exist_ok=True)
    return True

def load_top_parameters():
    """Load top parameters identified during exploration phase"""
    # Check if top parameters file exists
    if not os.path.exists(TOP_PARAMS_FILE):
        print(f"WARNING: Top parameters file not found at {TOP_PARAMS_FILE}")
        print("Will use all available parameters instead.")
        return None
    
    # Load parameters from file
    top_params = []
    with open(TOP_PARAMS_FILE, 'r') as f:
        for line in f.readlines()[1:]:  # Skip header line
            if line.strip():
                # Extract parameter name from line (format: "1. param_name: value")
                param = line.split('.', 1)[1].split(':', 1)[0].strip()
                top_params.append(param)
    
    return top_params

def load_accident_data(accident_type, top_params=None):
    """Load and preprocess data for a specific accident type"""
    accident_dir = os.path.join(DATA_DIR, accident_type)
    csv_files = [f for f in os.listdir(accident_dir) if f.endswith('.csv')]
    
    all_data = []
    file_ids = []
    
    for file in csv_files:
        try:
            # Extract file ID (number) for later reference
            file_id = os.path.splitext(file)[0]
            
            # Load the CSV file with robust error handling
            try:
                # Try standard engine first
                df = pd.read_csv(os.path.join(accident_dir, file))
            except Exception as e:
                try:
                    # If standard engine fails, try python engine
                    print(f"Retrying {file} with python engine...")
                    df = pd.read_csv(os.path.join(accident_dir, file), engine='python')
                except Exception as inner_e:
                    print(f"Error loading {file} from {accident_type}: {inner_e}")
                    continue
            
            # If top parameters are specified, use only those
            if top_params is not None:
                # Keep only parameters that exist in the dataset
                valid_params = [p for p in top_params if p in df.columns]
                if len(valid_params) < len(top_params):
                    missing = set(top_params) - set(valid_params)
                    print(f"WARNING: Missing {len(missing)} parameters in {file} from {accident_type}")
                
                df = df[valid_params]
            
            # Handle missing values using modern syntax
            df = df.ffill().bfill()
            
            # If there are still NaNs, replace with zeros
            df = df.fillna(0)
            
            all_data.append(df.values)
            file_ids.append(file_id)
            
        except Exception as e:
            print(f"Error loading {file} from {accident_type}: {e}")
    
    return all_data, file_ids

def create_windows(data, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """Create fixed-length windows with overlap from time-series data"""
    windows = []
    
    for sample in data:
        # Check if sample has enough time steps
        if len(sample) < window_size:
            # If sample is too short, pad with zeros or skip
            print(f"WARNING: Sample with {len(sample)} time steps is shorter than window size {window_size}")
            # For very short samples, we could pad with zeros, but we'll skip for simplicity
            if len(sample) < window_size / 2:
                continue
            
            # Pad with zeros if at least half the window size
            padding = np.zeros((window_size - len(sample), sample.shape[1]))
            padded_sample = np.vstack([sample, padding])
            windows.append(padded_sample)
            continue
        
        # For normal-sized samples, create overlapping windows
        step = window_size - overlap
        
        # Calculate how many windows we can create
        num_windows = (len(sample) - window_size) // step + 1
        
        # Create windows
        for i in range(num_windows):
            start_idx = i * step
            end_idx = start_idx + window_size
            window = sample[start_idx:end_idx]
            windows.append(window)
    
    return np.array(windows)

def preprocess_all_data(top_params=None):
    """Preprocess data for all accident types"""
    X_data = []  # Features
    y_data = []  # Labels (accident types)
    metadata = []  # Metadata about each window (accident type, file id, etc.)
    
    # Loop through each accident type
    for accident_idx, accident_type in enumerate(ACCIDENT_TYPES):
        print(f"Processing {accident_type}...")
        
        # Load data for this accident type
        accident_data, file_ids = load_accident_data(accident_type, top_params)
        
        if not accident_data:
            print(f"WARNING: No data loaded for {accident_type}")
            continue
        
        # Create windows for each sample
        for sample_idx, (sample, file_id) in enumerate(zip(accident_data, file_ids)):
            windows = create_windows([sample])
            
            if len(windows) > 0:
                X_data.append(windows)
                
                # Create labels (one-hot not needed here, will be done later)
                y_data.append(np.full(len(windows), accident_idx))
                
                # Add metadata
                for i in range(len(windows)):
                    metadata.append({
                        'accident_type': accident_type,
                        'file_id': file_id,
                        'window_index': i,
                        'label_index': accident_idx
                    })
    
    # Concatenate all windows and labels
    X_data = np.vstack(X_data)
    y_data = np.concatenate(y_data)
    
    print(f"Created {len(X_data)} windows from all accident types")
    print(f"Features shape: {X_data.shape}")
    print(f"Labels shape: {y_data.shape}")
    
    return X_data, y_data, metadata

def normalize_data(X_train, X_val, X_test):
    """Normalize data using StandardScaler"""
    # Reshape data to 2D for scaling (samples*timesteps, features)
    n_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_train * n_timesteps, n_features)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    
    # Apply scaling to all datasets
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(n_train, n_timesteps, n_features)
    
    # Apply same scaling to validation and test data
    if X_val is not None:
        n_val = X_val.shape[0]
        X_val_reshaped = X_val.reshape(n_val * n_timesteps, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_val, n_timesteps, n_features)
    else:
        X_val_scaled = None
    
    if X_test is not None:
        n_test = X_test.shape[0]
        X_test_reshaped = X_test.reshape(n_test * n_timesteps, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_test, n_timesteps, n_features)
    else:
        X_test_scaled = None
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def split_data(X, y, metadata, test_size=0.15, val_size=0.15):
    """Split data into training, validation, and test sets"""
    # First split: training and temp (validation + test)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, np.arange(len(X)), 
        test_size=test_size+val_size, 
        random_state=42, 
        stratify=y
    )
    
    # Second split: validation and test
    test_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=test_ratio, 
        random_state=42, 
        stratify=y_temp
    )
    
    # Get metadata for each split
    meta_train = [metadata[i] for i in idx_train]
    meta_val = [metadata[i] for i in idx_val]
    meta_test = [metadata[i] for i in idx_test]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, 
                        meta_train, meta_val, meta_test, scaler, top_params=None):
    """Save processed data to files"""
    # Save the datasets
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    
    # Save metadata
    with open(os.path.join(PROCESSED_DIR, 'meta_train.json'), 'w') as f:
        json.dump(meta_train, f)
    
    with open(os.path.join(PROCESSED_DIR, 'meta_val.json'), 'w') as f:
        json.dump(meta_val, f)
    
    with open(os.path.join(PROCESSED_DIR, 'meta_test.json'), 'w') as f:
        json.dump(meta_test, f)
    
    # Save scaler
    with open(os.path.join(PROCESSED_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save preprocessing info
    preprocessing_info = {
        'accident_types': ACCIDENT_TYPES,
        'window_size': WINDOW_SIZE,
        'overlap': OVERLAP,
        'top_parameters': top_params if top_params else "all",
        'train_size': len(y_train),
        'val_size': len(y_val),
        'test_size': len(y_test),
        'feature_count': X_train.shape[2],
        'label_mapping': {i: label for i, label in enumerate(ACCIDENT_TYPES)}
    }
    
    with open(os.path.join(PROCESSED_DIR, 'preprocessing_info.json'), 'w') as f:
        json.dump(preprocessing_info, f, indent=4)
    
    print(f"Saved processed data to {PROCESSED_DIR}")

def main():
    """Main function to run the data preprocessing pipeline"""
    print("Starting Nuclear Accident Data Preprocessing")
    print("=" * 80)
    
    # Step 1: Create output directory
    create_output_directory()
    
    # Step 2: Load top parameters from exploration phase
    top_params = load_top_parameters()
    if top_params:
        print(f"Using {len(top_params)} top parameters identified during exploration")
    else:
        print("Using all available parameters (no parameter selection)")
    
    # Step 3: Preprocess all data
    X, y, metadata = preprocess_all_data(top_params)
    
    # Step 4: Split data into train, validation, and test sets
    (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test) = split_data(X, y, metadata)
    
    # Step 5: Normalize data
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(X_train, X_val, X_test)
    
    # Step 6: Save processed data
    save_processed_data(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
        meta_train, meta_val, meta_test, scaler, top_params
    )
    
    print("\nPreprocessing complete!")
    print("=" * 80)
    
    # Return important info for next steps
    return {
        'X_train_shape': X_train_scaled.shape,
        'y_train_shape': y_train.shape,
        'X_val_shape': X_val_scaled.shape,
        'y_val_shape': y_val.shape,
        'X_test_shape': X_test_scaled.shape,
        'y_test_shape': y_test.shape,
        'num_features': X_train_scaled.shape[2],
        'num_classes': len(np.unique(y_train))
    }

if __name__ == "__main__":
    main() 