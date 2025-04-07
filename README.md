# Nuclear Accident Classification System

A deep learning-based system for classifying nuclear power plant accidents using time-series data. This project implements a hybrid LSTM-GRU model with attention mechanism for accurate accident classification.

## Project Structure

```
nuclear_accident_classifier/
├── data/
│   ├── LOCA/
│   ├── SGBTR/
│   ├── LR/
│   ├── MD/
│   ├── SGATR/
│   ├── SLBIC/
│   ├── LOCAC/
│   ├── RI/
│   ├── FLB/
│   ├── LLB/
│   ├── SLBOC/
│   └── RW/
├── scripts/
│   ├── data_exploration.py
│   └── data_preprocessing.py
├── lstm_model/
│   ├── model.py
│   └── train.py
├── processed_data/
├── requirements.txt
└── README.md
```

## Features

- Hybrid LSTM-GRU architecture with attention mechanism
- Bidirectional layers for better feature extraction
- Dropout regularization to prevent overfitting
- Early stopping and model checkpointing
- Comprehensive evaluation metrics and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nuclear_accident_classifier.git
cd nuclear_accident_classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Exploration:
```bash
python scripts/data_exploration.py
```

2. Data Preprocessing:
```bash
python scripts/data_preprocessing.py
```

3. Model Training:
```bash
python lstm_model/train.py
```

## Model Architecture

The model consists of:
- Bidirectional LSTM layers (2 layers)
- Bidirectional GRU layers (2 layers)
- Attention mechanism
- Dropout regularization
- Final classification layers

## Results

The model generates:
- Training and validation loss curves
- Confusion matrix
- Detailed classification report

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.2
- Matplotlib >= 3.4.2
- Seaborn >= 0.11.1

## License

This project is licensed under the MIT License - see the LICENSE file for details. 