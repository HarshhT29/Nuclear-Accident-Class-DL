# LSTM-GRU Model Implementation Explanation

## Overview
This document explains the implementation of a hybrid LSTM-GRU (Long Short-Term Memory - Gated Recurrent Unit) model for nuclear accident classification. The model is designed to process time-series data from nuclear power plant parameters and classify different types of accidents.

## Model Architecture

### 1. Input Layer
- Takes time-series data with shape `(sequence_length, n_features)`
- Each sequence represents a window of parameter measurements
- Features are preprocessed and normalized values from the nuclear plant parameters

### 2. Advanced LSTM-GRU Hybrid Architecture
The model uses a sophisticated combination of bidirectional LSTM and GRU layers with attention mechanisms and regularization:

#### First Bidirectional LSTM Layer
```python
layers.Bidirectional(
    layers.LSTM(160, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
)(inputs)
layers.BatchNormalization()
layers.Dropout(0.35)
```
- 160 units with bidirectional processing (total 320 units)
- L2 regularization to prevent overfitting
- Captures long-term dependencies from both past and future states
- Higher dropout rate (0.35) for improved regularization
- Batch normalization for training stability

#### Bidirectional GRU Layer
```python
layers.Bidirectional(
    layers.GRU(96, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
)(x)
layers.BatchNormalization()
layers.Dropout(0.35)
```
- 96 units with bidirectional processing (total 192 units)
- Efficiently captures medium-term dependencies
- L2 regularization for weight sparsity
- Further dropout and normalization for robust training

#### Attention Mechanism
```python
attention = layers.Dense(1, activation='tanh')(x)
attention = layers.Flatten()(attention)
attention = layers.Activation('softmax')(attention)
attention = layers.RepeatVector(192)(attention)
attention = layers.Permute([2, 1])(attention)
x = layers.Multiply()([x, attention])
x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
```
- Learns to focus on the most relevant time steps
- Applies attention weights to GRU outputs
- Improves model interpretability
- Enhances classification performance on complex sequences

#### Deep Feed-Forward Layers
```python
x = layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.35)(x)

x = layers.Dense(48, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)
```
- Deeper network with gradually decreasing dimensions
- L2 regularization applied throughout
- Multiple batch normalization layers
- Carefully tuned dropout rates for each layer

### 3. Advanced Model Compilation
```python
# Custom learning rate schedule with warmup
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    alpha=0.1
)

# Compile with Adam optimizer and gradient clipping
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, 
        clipnorm=1.0
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```
- Cosine decay learning rate schedule for better convergence
- Gradient clipping to prevent exploding gradients
- Advanced Adam optimizer configuration
- Maintains sparse categorical crossentropy for efficient training

## Training Process

### 1. Advanced Data Loading
- Loads preprocessed data from `processed_data/` directory
- Splits into training, validation, and test sets
- Maintains data shapes and normalization

### 2. Enhanced Training Configuration
```python
callbacks = [
    # Early stopping with more patience
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    # Model checkpoint for best validation accuracy
    ModelCheckpoint(
        filepath=os.path.join(self.model_path, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Reduce learning rate when progress stalls
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
]
```
- More sophisticated early stopping with increased patience
- Improved model checkpointing
- Dynamic learning rate reduction
- TensorBoard integration for monitoring
- Verbose progress tracking

### 3. Class Weight Balancing
```python
# Calculate class weights
class_weights = {}
total_samples = len(y_train)
n_classes = self.num_classes

for i in range(n_classes):
    class_count = np.sum(y_train == i)
    if class_count > 0:  # Avoid division by zero
        # Balanced weighting formula
        class_weights[i] = total_samples / (n_classes * class_count)
    else:
        class_weights[i] = 1.0
```
- Handles class imbalance automatically
- Gives more weight to underrepresented accident types
- Improves performance on rare accident scenarios

### 4. Training Execution
```python
history = self.model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)
```
- Leverages class weights for balanced training
- Uses all advanced callbacks
- Handles up to 100 epochs with intelligent early stopping

## Comprehensive Evaluation

### 1. Enhanced Model Evaluation
- Generates predictions on test set
- Calculates multiple classification metrics:
  - Overall accuracy
  - Weighted and macro F1 scores
  - Precision and recall metrics
  - Per-class performance statistics
- Creates standard and normalized confusion matrices

### 2. Advanced Visualization Tools
- Training history plots (accuracy and loss)
- Confusion matrix heatmaps (raw and normalized)
- ROC curves for each accident type
- Precision-recall curves
- Class distribution visualization

### 3. Detailed Output Analysis
- Per-class metrics (precision, recall, F1, specificity)
- Support values for each class
- Global and per-class performance summaries
- Saved prediction probabilities for further analysis

## Model Saving and Loading

### 1. Comprehensive Model Saving
```python
# Save model weights
self.model.save(os.path.join(self.model_path, 'model.h5'))

# Save evaluation metrics
with open(os.path.join(self.model_path, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# Save prediction probabilities for further analysis
np.save(os.path.join(self.model_path, 'y_pred_proba.npy'), y_pred_proba)
np.save(os.path.join(self.model_path, 'y_pred_classes.npy'), y_pred_classes)
np.save(os.path.join(self.model_path, 'y_test.npy'), y_test)
```
- Saves model weights and configuration
- Stores comprehensive evaluation metrics in JSON format
- Preserves prediction probabilities for advanced analysis
- Organized directory structure for all artifacts

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python lstm_model/train.py
```

3. Monitor training progress:
- View real-time training metrics
- Check TensorBoard logs
- Review generated visualizations
- Analyze comprehensive metrics

## Dependencies
- numpy >= 1.19.2
- tensorflow >= 2.8.0
- scikit-learn >= 0.24.2
- matplotlib >= 3.4.3
- seaborn >= 0.11.2
- pandas >= 1.3.0

## Notes
- The model uses a sophisticated bidirectional architecture with attention
- Class weight balancing addresses dataset imbalance
- Cosine decay learning rate schedule improves convergence
- Comprehensive evaluation provides detailed performance insights
- L2 regularization and higher dropout rates prevent overfitting 