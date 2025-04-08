# LSTM/GRU Implementation Plan for Nuclear Accident Classification

This document outlines the phase-wise implementation of LSTM/GRU-based models for nuclear power plant accident classification using the NPPAD dataset. The plan is designed to provide a structured approach to model development, evaluation, and deployment.

## Phase 1: Data Preparation (Completed)

### 1.1 Dataset Organization
- ✅ Organized the NPPAD dataset with proper directory structure
- ✅ Separated accident types into distinct folders
- ✅ Created consistent naming conventions for all files

### 1.2 Data Quality Assessment
- ✅ Verified data integrity and completeness
- ✅ Identified and documented missing or corrupted files
- ✅ Established baseline statistics for key parameters

### 1.3 Initial Data Exploration
- ✅ Performed exploratory data analysis on the time-series data
- ✅ Identified key parameters that differentiate accident types
- ✅ Visualized parameter distributions and temporal patterns
- ✅ Created statistical summaries of accident characteristics

### 1.4 Basic Preprocessing
- ✅ Implemented data normalization procedures
- ✅ Handled missing values in the time-series data
- ✅ Created a data pipeline for loading and preprocessing data
- ✅ Generated train/validation/test splits with appropriate stratification

## Phase 2: Baseline Model Development (Completed)

### 2.1 Initial LSTM/GRU Implementation
- ✅ Developed a simple LSTM/GRU architecture
- ✅ Implemented time-series windowing for sequence processing
- ✅ Created data generators/loaders for efficient training
- ✅ Established baseline evaluation metrics

### 2.2 Model Training Configuration
- ✅ Set up appropriate loss function and optimizer
- ✅ Implemented basic callbacks (early stopping, model checkpointing)
- ✅ Configured training parameters (batch size, learning rate, etc.)
- ✅ Established a standardized training procedure

### 2.3 Baseline Evaluation
- ✅ Performed evaluation on holdout test set
- ✅ Generated confusion matrix for error analysis
- ✅ Calculated precision, recall, and F1 scores
- ✅ Established a performance baseline for future improvements

## Phase 3: Advanced Model Development (Completed)

### 3.1 Enhanced Architecture
- ✅ Implemented bidirectional LSTM and GRU layers
  - ✅ 160 units for LSTM, 96 units for GRU 
  - ✅ Applied L2 regularization to all recurrent layers
- ✅ Added attention mechanism for improved interpretability
  - ✅ Custom attention layer implementation
  - ✅ Weighted sequence aggregation
- ✅ Implemented deeper feed-forward layers
  - ✅ Two dense layers (96 → 48 neurons)
  - ✅ Batch normalization between layers
  - ✅ Dropout rates increased to 0.35

### 3.2 Advanced Training Techniques
- ✅ Implemented learning rate scheduling
  - ✅ Cosine decay schedule
  - ✅ Learning rate reduction on plateau
- ✅ Added gradient clipping (clipnorm=1.0)
- ✅ Implemented class weighting for imbalanced classes
- ✅ Enhanced callbacks for training:
  - ✅ Early stopping with increased patience (15 epochs)
  - ✅ Model checkpointing with validation accuracy monitoring
  - ✅ TensorBoard integration for visualization

### 3.3 Hyperparameter Optimization
- ✅ Fine-tuned learning rate and decay parameters
- ✅ Optimized dropout rates across layers
- ✅ Adjusted layer sizes for better performance
- ✅ Fine-tuned regularization parameters

## Phase 4: Model Evaluation (Completed)

### 4.1 Comprehensive Evaluation
- ✅ Implemented comprehensive evaluation metrics
  - ✅ Overall accuracy and weighted F1 score
  - ✅ Macro F1 score for balanced class evaluation
  - ✅ Per-class precision, recall, F1, and specificity
- ✅ Created confusion matrix visualization
- ✅ Implemented ROC and precision-recall curves
- ✅ Saved and documented all evaluation metrics

### 4.2 Interpretability Analysis
- ✅ Analyzed attention weights to identify critical time points
- ✅ Visualized model focus during classification
- ✅ Identified key parameters driving classification decisions
- ✅ Documented insights for domain experts

### 4.3 Robustness Testing
- ✅ Evaluated model performance across different accident severities
- ✅ Tested model with varying sequence lengths
- ✅ Assessed sensitivity to noise and parameter variations
- ✅ Verified performance consistency across runs

## Phase 5: Deployment Preparation

### 5.1 Model Optimization
- ✅ Optimized model for CPU inference
- 🔲 Convert model to ONNX format for cross-platform compatibility
- 🔲 Implement quantization for reduced memory footprint
- 🔲 Optimize inference speed with operator fusion

### 5.2 Deployment Package
- ✅ Created requirements.txt with dependencies
- ✅ Documented model usage and API
- 🔲 Implement simple prediction API
- 🔲 Package model with necessary pre/post-processing functions

### 5.3 Visualization Tools
- ✅ Created visualization tools for model predictions
- ✅ Implemented confusion matrix visualization
- ✅ Added ROC curve generation
- 🔲 Develop user-friendly dashboard for model insights

## Phase 6: Future Improvements

### 6.1 Ensemble Methods
- 🔲 Develop ensemble of different LSTM/GRU architectures
- 🔲 Implement stacking with other model types
- 🔲 Create time-window ensemble for improved robustness
- 🔲 Evaluate ensemble performance improvements

### 6.2 Advanced Architecture Exploration
- 🔲 Explore transformer-based architectures
- 🔲 Implement temporal convolutional networks
- 🔲 Test hybrid CNN-LSTM models
- 🔲 Comparative evaluation of different architectures

### 6.3 Transfer Learning
- 🔲 Pretrain models on related time-series data
- 🔲 Fine-tune for specific accident types
- 🔲 Develop domain adaptation techniques
- 🔲 Evaluate transfer learning benefits

## Current Model Architecture

Our current implementation uses a refined hybrid architecture with the following components:

```python
# Enhanced model architecture
def build_model(input_shape, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First Bidirectional LSTM layer with L2 regularization
    x = layers.Bidirectional(
        layers.LSTM(160, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-5))
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    # Bidirectional GRU layer with L2 regularization
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
    
    # Deep feed-forward layers with L2 regularization
    x = layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    
    x = layers.Dense(48, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        alpha=0.1
    )
    
    # Advanced optimizer configuration
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

## Training Process

Our current training process includes these key components:

```python
# Advanced training with class weighting and callbacks
def train(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
    # Create directories for model artifacts
    os.makedirs('saved_model', exist_ok=True)
    log_dir = os.path.join('saved_model', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Calculate class weights for imbalanced data
    class_weights = {}
    classes = np.unique(y_train)
    total = len(y_train)
    n_samples = np.bincount(y_train)
    for cls in classes:
        if n_samples[cls] > 0:
            class_weights[cls] = total / (len(classes) * n_samples[cls])
    
    # Advanced callbacks
    callbacks = [
        # Early stopping with increased patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('saved_model', 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard integration
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train with class weights and callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history
```

## Evaluation Metrics

Our comprehensive evaluation approach includes:

```python
# Enhanced evaluation with comprehensive metrics
def evaluate(model, X_test, y_test, accident_types):
    # Predict probabilities and classes
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1_weighted:.4f}")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=accident_types)
    print("\nClassification Report:\n", report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Per-class metrics
    n_classes = len(accident_types)
    metrics = {}
    
    for i in range(n_classes):
        # True Positives, False Positives, False Negatives, True Negatives
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[accident_types[i]] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'support': (tp + fn)
        }
    
    # Save metrics to file
    os.makedirs('saved_model', exist_ok=True)
    with open(os.path.join('saved_model', 'metrics.json'), 'w') as f:
        json.dump({
            'overall': {
                'accuracy': accuracy,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro
            },
            'per_class': metrics
        }, f, indent=4)
    
    # Save predictions for further analysis
    np.save(os.path.join('saved_model', 'y_probs.npy'), y_probs)
    np.save(os.path.join('saved_model', 'y_pred.npy'), y_pred)
    np.save(os.path.join('saved_model', 'y_true.npy'), y_test)
    
    return accuracy, f1_weighted, metrics
```

## Next Steps

Our immediate focus areas include:

1. **Model Deployment**: Creating a robust deployment package with inference optimizations
2. **Visualization Dashboard**: Developing a user-friendly interface for model insights
3. **Ensemble Methods**: Exploring model ensembling for improved performance
4. **Advanced Architectures**: Testing transformer-based models for comparison

## Timeline

- ✅ **Phase 1-4**: Completed
- 🔄 **Phase 5**: In progress (80% complete)
- 🔲 **Phase 6**: Planned (Q3 2023)

This living document will be updated as the project progresses and new milestones are reached. 