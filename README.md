# Nuclear Power Plant Accident Classification

This repository contains a machine learning project for classifying nuclear power plant accident types using time-series operational parameter data from the NPPAD dataset.

## Project Overview

Nuclear power plants require robust accident detection and classification systems to ensure safety and enable rapid response. This project develops a classification model that can identify 12 different accident types based on time-series operational parameter data.

### Dataset

The Nuclear Power Plant Accident Dataset (NPPAD) contains:
- 12 accident types with 100 simulations per type
- Each simulation includes time-series data with ~97 operational parameters
- Data collected at 10-second intervals
- Parameters include pressure, temperature, flow rates, etc.

## Implementation Plan

The project follows a structured implementation plan:

1. **Data Exploration**: Understanding the dataset characteristics
2. **Data Preprocessing**: Normalizing parameters, handling missing values
3. **Model Development**: Building and refining the model architecture
4. **Model Evaluation**: Comprehensive performance assessment
5. **Documentation**: Clear explanation of implementation details

For a detailed implementation plan, see [lstm_gru_implementation_plan.md](lstm_gru_implementation_plan.md)

## Project Structure

```
project/
├── data_exploration/         # Scripts for exploring the dataset
│   └── explore_data.py       # Data exploration script
├── data_preprocessing/       # Scripts for preprocessing the data
│   └── preprocess_data.py    # Data preprocessing script
├── lstm_model/               # LSTM/GRU model implementation
│   ├── model.py              # Model architecture definition
│   ├── train.py              # Model training script
│   └── saved_model/          # Directory for saved models and results
├── docs/                     # Documentation
│   ├── data_exploration_explanation.md    # Data exploration details
│   ├── data_preprocessing_explanation.md  # Preprocessing details
│   └── lstm_model_explanation.md          # Model implementation details
├── NPPAD/                    # Dataset directory (not included in repo)
└── README.md                 # Project overview
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/username/nuclear-accident-classification.git
   cd nuclear-accident-classification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the NPPAD dataset and place it in the 'NPPAD' directory

## Usage

### Data Exploration

To explore the dataset characteristics:
```
python data_exploration/explore_data.py
```

### Data Preprocessing

To preprocess the raw data:
```
python data_preprocessing/preprocess_data.py
```

### Model Training

To train the enhanced bidirectional LSTM-GRU model:
```
python lstm_model/train.py
```

The script will:
1. Load preprocessed data
2. Build the hybrid model with bidirectional layers and attention
3. Train with automatic class weighting and advanced callbacks
4. Evaluate performance with comprehensive metrics
5. Save model artifacts and visualizations to `lstm_model/saved_model/`

## Model Architecture

Our current implementation features an enhanced hybrid architecture:

### Key Components

- **Bidirectional Layers**: Captures context from both past and future states
  - Bidirectional LSTM (160 units) with L2 regularization
  - Bidirectional GRU (96 units) with L2 regularization

- **Attention Mechanism**: Custom attention layer that focuses on critical time points

- **Advanced Regularization**:
  - L2 weight regularization
  - Increased dropout rates (0.35)
  - Batch normalization

- **Learning Rate Scheduling**:
  - Cosine decay schedule
  - Learning rate reduction on plateau

- **Class Imbalance Handling**:
  - Automatic class weighting based on sample distribution

### Training Process

- Early stopping with increased patience (15 epochs)
- Model checkpointing based on validation accuracy
- TensorBoard integration for visualization
- Gradient clipping to prevent exploding gradients

### Evaluation Metrics

The model is evaluated using:
- Overall accuracy
- Weighted and macro F1 scores
- Per-class precision, recall, and F1 scores
- Normalized confusion matrix
- ROC and precision-recall curves

## Project Goals

1. Develop a working prototype for nuclear accident classification
2. Achieve high classification accuracy across all accident types
3. Provide interpretable model insights through attention visualization
4. Create a foundation for future development of more advanced models

## Future Improvements

1. **Ensemble Methods**: Combining multiple model architectures
2. **Transfer Learning**: Pretraining on related time-series data
3. **Advanced Architectures**: Testing transformer-based models
4. **Deployment Pipeline**: Creating a robust inference system
5. **Interpretability Enhancements**: Advanced attention visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The NPPAD dataset providers
- Contributors to this project
- Nuclear power plant safety research community

## Data Preprocessing

The data preprocessing pipeline includes:

1. **Normalization**: Z-score normalization of all parameters
2. **Missing Value Handling**: Forward fill followed by zero imputation
3. **Sequence Windowing**: Fixed-length windows with appropriate overlap
4. **Train/Val/Test Split**: 70% training, 15% validation, 15% test with stratification
5. **One-Hot Encoding**: For categorical variables

For detailed information on preprocessing steps, see [data_preprocessing_explanation.md](docs/data_preprocessing_explanation.md). 