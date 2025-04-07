#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick data exploration script for nuclear accident classification
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
DATA_DIR = "NPPAD"  # Adjust if your dataset is located elsewhere
ACCIDENT_TYPES = ['LOCA', 'SGBTR', 'LR', 'MD', 'SGATR', 'SLBIC',
                     'LOCAC', 'RI', 'FLB', 'LLB', 'SLBOC','RW']  # Add all 12 accident types
NUM_SAMPLES_PER_TYPE = 10  # Using 10 samples for quick exploration
OUTPUT_DIR = "exploration_results"

def create_directory_structure():
    """Create necessary directories for the project"""
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Verify NPPAD data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"WARNING: Data directory {DATA_DIR} not found!")
        return False
    
    # Check if each accident type folder exists
    missing_folders = []
    for accident_type in ACCIDENT_TYPES:
        if not os.path.exists(os.path.join(DATA_DIR, accident_type)):
            missing_folders.append(accident_type)
    
    if missing_folders:
        print(f"WARNING: The following accident type folders are missing: {', '.join(missing_folders)}")
        return False
    
    return True

def check_data_integrity():
    """Perform quick sampling checks to verify data integrity"""
    integrity_results = {}
    
    for accident_type in ACCIDENT_TYPES:
        accident_dir = os.path.join(DATA_DIR, accident_type)
        csv_files = [f for f in os.listdir(accident_dir) if f.endswith('.csv')]
        txt_files = [f for f in os.listdir(accident_dir) if f.endswith('Transient Report.txt')]
        
        # Check if we have the expected number of files
        csv_count = len(csv_files)
        txt_count = len(txt_files)
        
        integrity_results[accident_type] = {
            'csv_files': csv_count,
            'txt_files': txt_count,
            'expected': 100,
            'csv_complete': csv_count == 100,
            'txt_complete': txt_count == 100
        }
        
        # Sample a few files to check format
        if csv_files:
            sample_csv = os.path.join(accident_dir, csv_files[0])
            try:
                df = pd.read_csv(sample_csv)
                integrity_results[accident_type]['sample_columns'] = df.shape[1]
                integrity_results[accident_type]['sample_rows'] = df.shape[0]
                integrity_results[accident_type]['csv_readable'] = True
            except Exception as e:
                integrity_results[accident_type]['csv_readable'] = False
                integrity_results[accident_type]['error'] = str(e)
    
    # Print integrity results
    print("\nData Integrity Check Results:")
    print("-" * 80)
    for accident_type, results in integrity_results.items():
        print(f"Accident Type: {accident_type}")
        print(f"  CSV Files: {results['csv_files']}/100 {'✓' if results['csv_complete'] else '✗'}")
        print(f"  TXT Files: {results['txt_files']}/100 {'✓' if results['txt_complete'] else '✗'}")
        if 'csv_readable' in results and results['csv_readable']:
            print(f"  Sample Data: {results['sample_rows']} rows × {results['sample_columns']} columns")
        elif 'error' in results:
            print(f"  Error: {results['error']}")
        print("-" * 80)
    
    return integrity_results

def load_sample_data():
    """Load a subset of data for rapid exploration"""
    all_data = {}
    
    for accident_type in ACCIDENT_TYPES:
        accident_dir = os.path.join(DATA_DIR, accident_type)
        csv_files = [f for f in os.listdir(accident_dir) if f.endswith('.csv')]
        
        # Select a subset of files
        selected_files = np.random.choice(csv_files, min(NUM_SAMPLES_PER_TYPE, len(csv_files)), replace=False)
        
        accident_data = []
        for file in selected_files:
            try:
                df = pd.read_csv(os.path.join(accident_dir, file))
                # Store the dataframe with file info
                accident_data.append({
                    'file': file,
                    'data': df
                })
            except Exception as e:
                print(f"Error loading {file} from {accident_type}: {e}")
        
        all_data[accident_type] = accident_data
    
    return all_data

def calculate_basic_statistics(all_data):
    """Calculate basic statistics for parameters across accident types"""
    # Collect parameter names from first available dataset
    first_accident = next(iter(all_data.values()))
    if not first_accident:
        print("No data available to calculate statistics")
        return {}
    
    params = first_accident[0]['data'].columns.tolist()
    
    # Initialize statistics dictionary
    stats = {param: {'mean': {}, 'std': {}, 'min': {}, 'max': {}} for param in params}
    
    # Calculate statistics for each parameter across accident types
    for accident_type, accident_data in all_data.items():
        if not accident_data:
            continue
            
        # Concatenate all dataframes for this accident type
        combined_df = pd.concat([item['data'] for item in accident_data])
        
        # Calculate statistics
        for param in params:
            if param in combined_df.columns:
                stats[param]['mean'][accident_type] = combined_df[param].mean()
                stats[param]['std'][accident_type] = combined_df[param].std()
                stats[param]['min'][accident_type] = combined_df[param].min()
                stats[param]['max'][accident_type] = combined_df[param].max()
    
    # Save statistics to CSV
    stats_df_list = []
    for param, param_stats in stats.items():
        for stat_type, stat_values in param_stats.items():
            for accident_type, value in stat_values.items():
                stats_df_list.append({
                    'parameter': param,
                    'statistic': stat_type,
                    'accident_type': accident_type,
                    'value': value
                })
    
    stats_df = pd.DataFrame(stats_df_list)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'parameter_statistics.csv'), index=False)
    
    return stats

def identify_distinctive_parameters(stats):
    """Identify parameters that show significant differences across accident types"""
    # Calculate the variance of means across accident types for each parameter
    parameter_variance = {}
    for param, param_stats in stats.items():
        mean_values = list(param_stats['mean'].values())
        if mean_values:
            # Calculate variance of the parameter means across accident types
            parameter_variance[param] = np.var(mean_values)
    
    # Sort parameters by variance (higher variance = more distinctive)
    sorted_params = sorted(parameter_variance.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 20-30 parameters (or fewer if there aren't that many)
    top_params = sorted_params[:min(30, len(sorted_params))]
    
    # Print and save top parameters
    print("\nTop Distinctive Parameters:")
    print("-" * 80)
    for param, variance in top_params:
        print(f"{param}: Variance = {variance:.6f}")
    
    # Save to file
    with open(os.path.join(OUTPUT_DIR, 'top_parameters.txt'), 'w') as f:
        f.write("Top Distinctive Parameters (by variance of means across accident types):\n")
        for i, (param, variance) in enumerate(top_params):
            f.write(f"{i+1}. {param}: {variance:.6f}\n")
    
    return [param for param, _ in top_params]

def create_visualizations(all_data, top_params):
    """Create visualizations of key parameters across different accident types"""
    # Select top 3 parameters for visualization
    vis_params = top_params[:3]
    
    for param in vis_params:
        plt.figure(figsize=(12, 8))
        
        for accident_type, accident_data in all_data.items():
            if not accident_data:
                continue
                
            # Get data for each sample
            for i, item in enumerate(accident_data):
                if param in item['data'].columns:
                    plt.plot(
                        item['data'][param], 
                        alpha=0.7,
                        label=f"{accident_type}_{i}" if i == 0 else "_nolegend_"
                    )
        
        plt.title(f"{param} Across Different Accident Types")
        plt.xlabel("Time Step")
        plt.ylabel(param)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{param}_comparison.png'))
        plt.close()
    
    # Create correlation heatmap for first sample of each accident type
    plt.figure(figsize=(15, 12))
    
    # Get first sample from each accident type
    sample_dfs = []
    accident_labels = []
    
    for accident_type, accident_data in all_data.items():
        if accident_data:
            sample_dfs.append(accident_data[0]['data'][top_params])
            accident_labels.append(accident_type)
    
    if sample_dfs:
        # Calculate correlation matrix
        corr_matrix = pd.concat(sample_dfs).corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Top Parameters')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
        plt.close()

def main():
    """Main function to run the data exploration process"""
    print("Starting Nuclear Accident Data Exploration")
    print("=" * 80)
    
    # Step 1: Create directory structure
    if not create_directory_structure():
        print("ERROR: Directory structure setup failed. Please check the data paths.")
        return
    
    # Step 2: Check data integrity
    integrity_results = check_data_integrity()
    
    # Step 3: Load sample data
    print("\nLoading sample data...")
    all_data = load_sample_data()
    
    # Step 4: Calculate basic statistics
    print("\nCalculating basic statistics...")
    stats = calculate_basic_statistics(all_data)
    
    # Step 5: Identify distinctive parameters
    print("\nIdentifying distinctive parameters...")
    top_params = identify_distinctive_parameters(stats)
    
    # Step 6: Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(all_data, top_params)
    
    print("\nExploration complete! Results saved to:", OUTPUT_DIR)
    print("=" * 80)
    
    # Return top parameters for use in preprocessing
    return top_params

if __name__ == "__main__":
    main() 