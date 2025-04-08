# One-Day LSTM/GRU Implementation Plan

This document outlines how to implement a basic LSTM/GRU model for nuclear accident classification in a single day, as well as our subsequent refinements for improved performance.

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

## Enhanced Model Implementation (Post One-Day)

Following our initial one-day implementation, we created a significantly enhanced model with the following improvements:

### Architecture Enhancements
1. **Bidirectional Processing**
   - Replaced standard LSTM/GRU layers with bidirectional variants
   - Increased unit counts (160 LSTM units, 96 GRU units)
   - Added bidirectional processing for both LSTM and GRU layers

2. **Attention Mechanism**
   - Implemented a custom attention layer to focus on critical time steps
   - Applied attention weights to capture the most relevant patterns
   - Added weighted sum approach for sequence aggregation

3. **Advanced Regularization**
   - Added L2 regularization to all recurrent and dense layers
   - Increased dropout rates to 0.35 for better generalization
   - Implemented batch normalization across all layers
   - Added gradient clipping to prevent exploding gradients

4. **Deeper Architecture**
   - Added an additional dense layer (96 → 48 → output)
   - Implemented residual-style connections where appropriate
   - Increased model capacity for better feature learning

### Training Enhancements
1. **Learning Rate Optimization**
   - Implemented cosine decay learning rate schedule
   - Added learning rate reduction on plateau
   - Fine-tuned initial learning rate and decay parameters

2. **Class Imbalance Handling**
   - Added automatic class weight calculation
   - Applied class weights during training
   - Balanced performance across all accident types

3. **Advanced Monitoring**
   - Added TensorBoard integration for visualization
   - Enhanced callback configuration with improved patience
   - Implemented more comprehensive model checkpointing

4. **Evaluation Improvements**
   - Added macro F1 score for balanced evaluation
   - Implemented per-class metrics for detailed analysis
   - Created normalized confusion matrix visualization
   - Added ROC and precision-recall curves for each class

### Performance Improvements
The enhanced model demonstrated significant improvements over the one-day implementation:
- Increased overall accuracy
- Better F1 scores, especially for underrepresented classes
- Improved generalization to test data
- More interpretable attention patterns
- Better stability during training

## Implementation Blueprint (Simplified Version)

```python
# Essential imports (original one-day version)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Quick data loading function - simplified for one-day implementation
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
    
    # For categorical loss
    y_data = tf.keras.utils.to_categorical(y_data, num_classes=len(accident_types))
    
    return X_data, y_data
```

## Enhanced Model Blueprint (Refined Version)

```python
# Advanced imports for enhanced implementation
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Enhanced model architecture with bidirectional layers and attention
def build_enhanced_model(input_shape, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(160, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    # Bidirectional GRU layer
    x = layers.Bidirectional(
        layers.GRU(96, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(192)(attention)  # 96*2 for bidirectional
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention to GRU output
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    
    # Deep feed-forward layers
    x = layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Dense(48, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        alpha=0.1
    )
    
    # Compile with advanced optimizer settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, 
            clipnorm=1.0
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

## Priority Features (One-Day Version)

1. Basic data loading and preprocessing
2. Simple LSTM/GRU model with minimal layers
3. Standard training procedure
4. Basic evaluation metrics
5. Quick documentation of results

## Additional Features (Enhanced Version)

1. Bidirectional recurrent layers
2. Custom attention mechanism
3. Advanced regularization techniques
4. Learning rate scheduling
5. Class weight balancing
6. Comprehensive evaluation metrics
7. Detailed visualization tools
8. Improved model interpretability

## Success Criteria (Enhanced Version)

The refined model implementation achieves:

1. Higher accuracy and F1 scores across all accident types
2. Better handling of class imbalance
3. More stable training with advanced learning rate scheduling
4. Enhanced interpretability through attention mechanisms
5. Detailed performance analysis with multiple metrics and visualizations

This document demonstrates both the rapid one-day prototype approach and our subsequent enhancements to create a more sophisticated and effective model. 