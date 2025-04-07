# LSTM-GRU Model Implementation Explanation

## Overview
This document explains the implementation of a hybrid LSTM-GRU (Long Short-Term Memory - Gated Recurrent Unit) model for nuclear accident classification. The model is designed to process time-series data from nuclear power plant parameters and classify different types of accidents.

## Model Architecture

### 1. Input Layer
- Takes time-series data with shape `(sequence_length, n_features)`
- Each sequence represents a window of parameter measurements
- Features are preprocessed and normalized values from the nuclear plant parameters

### 2. LSTM-GRU Hybrid Architecture
The model uses a combination of LSTM and GRU layers to capture both long-term and short-term dependencies:

#### First LSTM Layer
```python
layers.LSTM(128, return_sequences=True, input_shape=self.input_shape)
layers.Dropout(0.3)
layers.BatchNormalization()
```
- 128 units with return sequences enabled
- Captures long-term dependencies in the data
- Dropout (0.3) prevents overfitting
- Batch normalization stabilizes training

#### GRU Layer
```python
layers.GRU(64, return_sequences=True)
layers.Dropout(0.3)
layers.BatchNormalization()
```
- 64 units with return sequences enabled
- Efficiently captures medium-term dependencies
- Dropout and batch normalization for regularization

#### Second LSTM Layer
```python
layers.LSTM(32)
layers.Dropout(0.3)
layers.BatchNormalization()
```
- 32 units without return sequences
- Final sequence processing layer
- Further dropout and normalization

#### Dense Layers
```python
layers.Dense(64, activation='relu')
layers.Dropout(0.3)
layers.Dense(self.num_classes, activation='softmax')
```
- Intermediate dense layer with ReLU activation
- Final classification layer with softmax activation
- Number of output classes equals number of accident types

### 3. Model Compilation
```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- Adam optimizer for adaptive learning rates
- Sparse categorical crossentropy loss for multi-class classification
- Accuracy metric for monitoring training progress

## Training Process

### 1. Data Loading
- Loads preprocessed data from `processed_data/` directory
- Splits into training, validation, and test sets
- Maintains data shapes and normalization

### 2. Training Configuration
```python
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]
```
- Early stopping prevents overfitting
- Model checkpointing saves best weights
- 100 epochs maximum with batch size of 32

### 3. Training Execution
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)
```
- Trains on training data
- Validates on validation set
- Monitors loss and accuracy

## Evaluation and Visualization

### 1. Model Evaluation
- Generates predictions on test set
- Calculates classification metrics
- Creates confusion matrix

### 2. Visualization Tools
- Training history plots (accuracy and loss)
- Confusion matrix heatmap
- Classification report with precision, recall, and F1-score

### 3. Saved Outputs
- Trained model weights
- Model configuration
- Training history plots
- Confusion matrix
- Classification report

## Model Saving and Loading

### 1. Saving Model
```python
model.save(os.path.join(self.model_path, 'model.h5'))
```
- Saves model weights
- Saves configuration
- Creates necessary directories

### 2. Loading Model
```python
@classmethod
def load_model(cls, model_path):
    # Load configuration and weights
    # Recreate model instance
```
- Loads saved configuration
- Recreates model architecture
- Loads saved weights

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Monitor training progress:
- Check training history plots
- Review classification report
- Analyze confusion matrix

## Dependencies
- numpy >= 1.19.2
- tensorflow >= 2.8.0
- scikit-learn >= 0.24.2
- matplotlib >= 3.4.3
- seaborn >= 0.11.2
- pandas >= 1.3.0

## Notes
- The model uses a hybrid architecture to capture different temporal dependencies
- Early stopping and dropout prevent overfitting
- Batch normalization stabilizes training
- The model saves checkpoints for the best performing version
- Comprehensive evaluation metrics help assess model performance 