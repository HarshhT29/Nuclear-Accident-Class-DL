# Nuclear Power Plant Accident Classification Models

## Problem Overview
Developing a classifier to identify nuclear power plant accident types using time-series operational parameter data from the NPPAD dataset.

## Dataset Description
- 12 accident types with 100 simulations per type (varying severity)
- Each simulation contains:
  - CSV file with ~97 operational parameters at 10-second intervals
  - Text report detailing subsystem actions during the simulation

## Proposed Models

### 1. LSTM/GRU Networks

**Description:** Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks are specialized recurrent neural networks designed to capture long-range dependencies in sequential data.

**Architecture:**
- Input layer (97 features)
- 1-3 LSTM/GRU layers (64-256 units each)
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

**Implementation Plan:**
1. Preprocess data:
   - Normalize/standardize all parameters
   - Handle missing values
   - Consider dimensionality reduction if needed
   - Segment time-series into appropriate windows
2. Build model architecture:
   - Experiment with both LSTM and GRU cells
   - Test bidirectional variants
   - Implement attention mechanisms to focus on critical time periods
3. Training approach:
   - Use cross-entropy loss
   - Implement early stopping
   - Try various optimizers (Adam, RMSprop)
   - Use learning rate scheduling
4. Validation:
   - k-fold cross-validation
   - Confusion matrix analysis
   - Class activation mapping to identify important parameters

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

**Implementation Plan:**
1. Preprocess data:
   - Normalize/standardize parameters
   - Segment time-series data appropriately
   - Consider spectral transformations (FFT features)
2. Build model architecture:
   - Start with simple architecture (3-5 conv layers)
   - Test different kernel sizes (3, 5, 7)
   - Implement residual connections for deeper networks
3. Training approach:
   - Test various optimizers
   - Implement data augmentation (adding noise, time warping)
   - Consider curriculum learning (easier accidents first)
4. Validation:
   - Monitor for overfitting
   - Analyze model attention to different time segments
   - Evaluate performance across accident severities

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

**Implementation Plan:**
1. Preprocess data:
   - Normalize all parameters
   - Consider sequence downsampling for efficiency
   - Robust handling of missing values
2. Build model architecture:
   - Adapt transformer encoder architecture for time-series
   - Test different attention mechanisms (multi-head, linear attention)
   - Experiment with various positional encodings
3. Training approach:
   - Implement efficient batching
   - Use learning rate warmup strategies
   - Consider transfer learning from pretrained time-series transformers
4. Validation:
   - Analyze attention maps for insights into critical parameters
   - Evaluate robustness to missing data
   - Test on varying sequence lengths

### 4. Hybrid CNN-LSTM Models

**Description:** Combines CNNs for feature extraction with LSTMs for temporal modeling, leveraging the strengths of both approaches.

**Architecture:**
- Input layer (97 features)
- 1D CNN layers for feature extraction
- LSTM/GRU layers for temporal dependencies
- Attention mechanism
- Dense layers
- Softmax output layer (12 classes)

**Pros:**
- Leverages strengths of both CNN and RNN architectures
- Better feature extraction than pure RNN models
- More efficient than pure transformers
- Handles both local patterns and long-term dependencies

**Cons:**
- More complex architecture to tune
- Can be computationally intensive
- Risk of overfitting with limited data

**Implementation Plan:**
1. Preprocess data:
   - Standard normalization procedures
   - Feature importance analysis to focus on critical parameters
2. Build model architecture:
   - Use CNNs for initial feature extraction
   - Pass extracted features to LSTM/GRU layers
   - Experiment with skip connections
3. Training approach:
   - Consider progressive training (train CNN first, then RNN)
   - Implement appropriate regularization
   - Use transfer learning where possible
4. Validation:
   - Comprehensive ablation studies
   - Performance analysis by accident type and severity
   - Interpretability analysis

### 5. Temporal Convolutional Networks (TCN)

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

**Implementation Plan:**
1. Preprocess data:
   - Standard normalization procedures
   - Consider multi-scale representations
2. Build model architecture:
   - Design appropriate dilation pattern
   - Implement residual connections
   - Add regularization layers
3. Training approach:
   - Standard optimization with learning rate scheduling
   - Implement data augmentation strategies
   - Experiment with different loss functions
4. Validation:
   - Compare with other architectures
   - Analyze performance across different time scales
   - Test robustness to noise

## Implementation Considerations

### Data Preprocessing
- Feature scaling (min-max or standardization)
- Missing value imputation
- Feature selection based on domain knowledge
- Time-series segmentation strategies
- Data augmentation techniques for nuclear data

### Training Approach
- Train on varying sequence lengths
- Consider multi-task learning (predict accident type and severity)
- Use of transfer learning where applicable
- Incorporate physics-informed constraints

### Evaluation Metrics
- Classification accuracy
- F1-score (important for imbalanced classes)
- Confusion matrix analysis
- Time-to-detection metrics
- Robustness to noise and missing data

### Explainability
- Feature importance analysis
- Critical time-point identification
- Integration with domain knowledge
- Uncertainty quantification

## Recommended First Approach

Based on the nature of your data, a **Hybrid CNN-LSTM model with attention** is recommended as an initial approach, balancing performance and computational requirements while maintaining interpretability.

For production deployment, ensemble methods combining multiple model types may provide the most robust performance across different accident scenarios and severity levels. 