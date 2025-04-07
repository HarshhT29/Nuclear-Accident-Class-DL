# LSTM-GRU Hybrid Model for Nuclear Accident Classification

This directory contains the implementation of a hybrid LSTM-GRU model with attention mechanism for nuclear accident classification.

## Model Architecture

The model combines the strengths of LSTM and GRU networks with an attention mechanism:

1. **LSTM Layers**:
   - Bidirectional LSTM with 2 layers
   - Hidden size: 128
   - Dropout: 0.2 (if multiple layers)
   - Returns sequences for GRU processing

2. **GRU Layers**:
   - Bidirectional GRU with 2 layers
   - Hidden size: 128
   - Dropout: 0.2 (if multiple layers)
   - Processes LSTM outputs

3. **Attention Mechanism**:
   - Learns to focus on important time steps
   - Uses softmax for attention weights
   - Produces weighted sum of GRU outputs

4. **Classification Layers**:
   - Dense layer with ReLU activation
   - Dropout (0.2)
   - Final dense layer for classification

## Training Process

The training script (`train.py`) implements:

1. **Data Loading**:
   - Loads preprocessed data from `processed_data/`
   - Creates PyTorch datasets and dataloaders
   - Batch size: 32

2. **Training Loop**:
   - 50 epochs
   - Adam optimizer (lr=0.001)
   - Cross-entropy loss
   - Early stopping based on validation loss
   - Model checkpointing

3. **Evaluation**:
   - Classification report
   - Confusion matrix
   - Loss curves visualization

## Usage

1. Ensure preprocessed data is available in `processed_data/`
2. Run the training script:
```bash
python train.py
```

## Output Files

The training process generates:
- `best_model.pth`: Best model weights
- `loss_curves.png`: Training and validation loss curves
- `confusion_matrix.png`: Confusion matrix visualization

## Model Parameters

Default parameters:
- Input dimension: 30 (features)
- Hidden dimension: 128
- Number of layers: 2
- Number of classes: 12
- Dropout rate: 0.2

## Performance Metrics

The model is evaluated using:
- Classification accuracy
- Precision, recall, and F1-score per class
- Confusion matrix for error analysis
- Training and validation loss curves

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Scikit-learn >= 0.24.2
- Matplotlib >= 3.4.2
- Seaborn >= 0.11.1 