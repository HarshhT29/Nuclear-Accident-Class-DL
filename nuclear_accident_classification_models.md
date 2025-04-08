# Nuclear Power Plant Accident Classification Models

## Problem Overview
Developing a classifier to identify nuclear power plant accident types using time-series operational parameter data from the NPPAD dataset.

## Dataset Description
- 12 accident types with 100 simulations per type (varying severity)
- Each simulation contains:
  - CSV file with ~97 operational parameters at 10-second intervals
  - Text report detailing subsystem actions during the simulation

## Implemented Model: Enhanced Bidirectional LSTM-GRU with Attention

Our refined implementation uses a sophisticated hybrid architecture with several advanced features:

### Architecture Details
- **Input Layer:** Takes time-series windows of plant parameters
- **First Layer:** Bidirectional LSTM (160 units) with L2 regularization
- **Second Layer:** Bidirectional GRU (96 units) with L2 regularization
- **Attention Mechanism:** Custom attention layer to focus on key time points
- **Feed-Forward Layers:** Multiple dense layers with batch normalization
- **Output Layer:** Softmax classification for 12 accident types

### Key Enhancements
- **Bidirectional Processing:** Captures context from both past and future states
- **Increased Model Capacity:** Larger layers (320 total LSTM units, 192 total GRU units)
- **Advanced Regularization:** L2 weight regularization, increased dropout rates (0.35)
- **Learning Rate Scheduling:** Cosine decay schedule for better convergence
- **Gradient Clipping:** Prevents exploding gradients during training
- **Class Weight Balancing:** Addresses dataset imbalance issues

### Training Approach
- **Automatic Class Weighting:** Compensates for underrepresented accident types
- **Advanced Callbacks:** Early stopping, learning rate reduction, TensorBoard
- **Comprehensive Evaluation:** Multiple metrics including per-class performance

### Performance Highlights
- **Weighted F1 Score:** Improved handling of class imbalance
- **Macro F1 Score:** Better performance across all accident types
- **Per-Class Metrics:** Detailed insights into model strengths and weaknesses
- **Improved Interpretability:** Attention maps highlight critical time points

## Alternative Models (For Future Exploration)

### 1. Pure LSTM Networks

**Description:** Long Short-Term Memory (LSTM) networks designed to capture long-range dependencies in sequential data.

**Architecture:**
- Input layer (97 features)
- 1-3 LSTM layers (64-256 units each)
- Dropout layers (0.2-0.5)
- Dense layers with ReLU activation
- Softmax output layer (12 classes)

**Pros:**
- Well-suited for time-series data with temporal dependencies
- Can capture long-term patterns across operational parameters
- Established track record in anomaly detection systems
- Relatively lightweight compared to transformers

**Cons:**
- Can suffer from vanishing gradient problems for very long sequences
- Training time increases significantly with sequence length
- May require careful sequence padding/truncation

### 2. 1D Convolutional Neural Networks

**Description:** 1D CNNs apply convolution operations across the time dimension to extract features from time-series data.

**Architecture:**
- Input layer (97 features)
- Multiple 1D convolutional layers with increasing filter sizes
- Max pooling layers
- Batch normalization
- Flattening layer
- Dense layers with ReLU activation
- Softmax output layer (12 classes)

**Pros:**
- Effective at capturing local patterns and features
- Faster training compared to recurrent networks
- Less susceptible to vanishing gradients
- Good performance with fixed-length inputs

**Cons:**
- May miss long-term dependencies without proper architectural design
- Less interpretable than some other models
- Requires careful kernel size selection

### 3. Transformer-Based Models

**Description:** Adapted from NLP, transformers use self-attention mechanisms to model relationships between all time steps in the sequence.

**Architecture:**
- Input embedding layer
- Positional encoding
- Multiple transformer encoder layers (self-attention + feed-forward)
- Global average pooling
- Dense layers
- Softmax output layer (12 classes)

**Pros:**
- Excels at capturing complex dependencies regardless of temporal distance
- Highly parallelizable (faster training on appropriate hardware)
- State-of-the-art performance on many sequence tasks
- Attention maps provide interpretability

**Cons:**
- Computationally expensive for long sequences
- Requires more data to train effectively
- More complex to implement and tune

### 4. Temporal Convolutional Networks (TCN)

**Description:** Specialized 1D CNNs with dilated convolutions that provide an exponentially large receptive field.

**Architecture:**
- Input layer (97 features)
- Multiple TCN blocks with increasing dilation rates
- Global pooling
- Dense layers
- Softmax output layer (12 classes)

**Pros:**
- Captures long-range dependencies with fewer parameters than RNNs
- Parallelizable (faster training)
- Stable gradients during training
- Flexible receptive field size

**Cons:**
- Less established than other approaches for this domain
- May require careful architecture design
- Less interpretable than attention-based models

## Model Selection Rationale

We chose the enhanced Bidirectional LSTM-GRU with Attention model for these key reasons:

1. **Superior Temporal Understanding:** The bidirectional architecture captures dependencies from both directions
2. **Attention for Interpretability:** The attention mechanism highlights critical time points in the accident progression
3. **Regularization for Robustness:** Comprehensive regularization prevents overfitting despite limited data
4. **Class Imbalance Handling:** Automatic class weighting improves performance across all accident types
5. **Enhanced Training Process:** Advanced callbacks and learning rate scheduling improve convergence

## Evaluation Metrics

Our comprehensive evaluation approach includes:

- **Accuracy:** Overall classification accuracy
- **F1 Score (Weighted):** Performance metric adjusted for class imbalance
- **F1 Score (Macro):** Equal consideration to all classes
- **Per-class Precision and Recall:** Detailed performance analysis
- **Confusion Matrix:** Visual representation of classification patterns
- **ROC and Precision-Recall Curves:** Detailed discrimination capabilities

## Future Improvements

Potential enhancements for the current model:

1. **Ensemble Approaches:** Combine multiple model architectures for improved robustness
2. **Domain-specific Feature Engineering:** Incorporate physics-based features
3. **Transfer Learning:** Pretrain on related time-series data
4. **Advanced Hyperparameter Tuning:** Bayesian optimization for parameter selection
5. **Interpretability Enhancements:** More advanced attention visualization tools 