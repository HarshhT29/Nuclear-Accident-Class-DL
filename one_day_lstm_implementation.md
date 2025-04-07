# One-Day LSTM/GRU Implementation Plan

This condensed plan outlines how to implement a basic LSTM/GRU model for nuclear accident classification in a single day.

## Morning (First 4 Hours)

### Hour 1: Quick Data Setup & Exploration
- Create minimal directory structure for NPPAD dataset
- Verify data integrity with quick sampling checks
- Load a subset of data for rapid exploration
- Calculate basic statistics for a few key parameters
- Create simple visualizations of 2-3 most distinctive parameters across accident types
- Identify the most predictive parameters (select top 20-30 instead of using all 97)

### Hour 2: Fast Data Preprocessing
- Apply standard normalization (z-score) to all selected parameters
- Handle missing values with forward fill (fast approach)
- Create fixed-length windows (e.g., 30 time steps) with minimal overlap
- Split data into 70% training, 15% validation, 15% test sets
- Create data generators/loaders for model training

### Hour 3: Baseline Model Implementation
- Implement a simple LSTM/GRU architecture:
  - Single LSTM/GRU layer (128 units)
  - Dropout (0.3)
  - Dense layer (64 units, ReLU activation)
  - Output layer (12 classes, softmax activation)
- Set up basic model compilation:
  - Adam optimizer (lr=0.001)
  - Categorical cross-entropy loss
  - Accuracy metric

### Hour 4: Initial Training & Validation
- Train the model for 10-20 epochs (or use early stopping)
- Evaluate on validation set
- Generate initial performance metrics
- Identify any obvious issues

## Afternoon (Next 4 Hours)

### Hour 5: Model Refinement
- Based on initial results, make one targeted improvement:
  - Add a second recurrent layer OR
  - Implement bidirectional version OR
  - Adjust learning rate OR
  - Change sequence length
- Retrain with the refinement

### Hour 6: Final Evaluation
- Evaluate refined model on test set
- Generate comprehensive metrics:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix
- Analyze performance per accident type

### Hour 7: Basic Interpretability
- Create simple visualizations to understand model decisions
- Identify which time steps are most important (if possible)
- Document any clear patterns in how the model classifies each accident type

### Hour 8: Documentation & Finalization
- Document the implementation process
- Create a simple notebook or script for end-to-end execution
- Summarize results and findings
- Outline next steps for future improvements

## Implementation Blueprint (Rapid Development)

```python
# Essential imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Quick data loading function
def load_data(accident_types, num_samples_per_type=50):
    X_data = []
    y_data = []
    
    for accident_idx, accident_type in enumerate(accident_types):
        # Load subset of data for each accident type
        for i in range(num_samples_per_type):
            try:
                # Adjust path as needed
                df = pd.read_csv(f"NPPAD/{accident_type}/{i}.csv")
                
                # Use only first 30 time steps and 30 key parameters for speed
                data_subset = df.iloc[:30, :30].values
                
                # Simple handling of missing values
                data_subset = np.nan_to_num(data_subset, nan=0)
                
                X_data.append(data_subset)
                y_data.append(accident_idx)
            except:
                continue
    
    # Convert to numpy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    # Normalize data quickly
    X_data = (X_data - np.mean(X_data, axis=0)) / np.std(X_data, axis=0)
    
    # One-hot encode labels
    y_data = tf.keras.utils.to_categorical(y_data, num_classes=len(accident_types))
    
    return X_data, y_data

# Build simple model
def build_quick_model(input_shape, num_classes):
    model = Sequential()
    
    # Choose either LSTM or GRU based on initial testing
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Fast training with early stopping
def quick_train(model, X_train, y_train, X_val, y_val):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    return model, history

# Main execution pipeline
def main():
    # Define accident types
    accident_types = ["LOCA", "SGTR", "MSLB", "etc."]  # Add all 12 types
    
    # Load and prepare data
    X_data, y_data = load_data(accident_types, num_samples_per_type=50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    # Build and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_quick_model(input_shape, len(accident_types))
    model, history = quick_train(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Print results
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(y_true, y_pred))
    
    # Save model
    model.save("quick_nuclear_accident_classifier.h5")
    
if __name__ == "__main__":
    main()
```

## Priority Features (Focus Only on These)

1. Basic data loading and preprocessing
2. Simple LSTM/GRU model with minimal layers
3. Standard training procedure
4. Basic evaluation metrics
5. Quick documentation of results

## What to Skip (Save for Later)

1. Advanced feature engineering
2. Hyperparameter optimization
3. Complex architectures (transformers, etc.)
4. Extensive interpretability analysis
5. Deployment pipeline
6. Attention mechanisms
7. Advanced regularization techniques

## Success Criteria (One-Day Version)

For a one-day implementation, success is defined as:

1. A working model that classifies accidents better than random guessing
2. Basic understanding of model performance across accident types
3. Documentation of the development process
4. Identification of clear next steps for improvement

Remember: The goal is a working prototype, not a production-ready system. 