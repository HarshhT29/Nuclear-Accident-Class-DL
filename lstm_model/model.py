import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class NuclearAccidentClassifier:
    def __init__(self, input_shape, num_classes, model_path='lstm_model/saved_model'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = model_path
        self.model = self._build_model()
        
    def _build_model(self):
        """Build an improved hybrid LSTM-GRU model with bidirectional layers and attention"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
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
        attention = layers.RepeatVector(192)(attention)  # 96*2 (bidirectional output dim)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention to GRU output
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Second LSTM layer (reduced size)
        x = layers.Reshape((1, 192))(x)
        x = layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        
        # Dense layers with increased depth
        x = layers.Dense(96, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        
        x = layers.Dense(48, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
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
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with advanced callbacks and techniques"""
        # Create directories
        os.makedirs(self.model_path, exist_ok=True)
        log_dir = os.path.join(self.model_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create callbacks
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
        
        # Train the model with class weights to handle imbalanced data
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
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model with comprehensive metrics and per-class analysis"""
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        f1_weighted = f1_score(y_test, y_pred_classes, average='weighted')
        f1_macro = f1_score(y_test, y_pred_classes, average='macro')
        
        # Calculate precision and recall
        precision_weighted = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        
        # Print overall metrics
        print("\nOverall Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")
        
        # Generate detailed classification report
        report = classification_report(y_test, y_pred_classes, zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Per-class metrics
        per_class_metrics = {}
        classes = np.unique(y_test)
        
        for cls in classes:
            # Binary mask for this class
            y_true_cls = (y_test == cls)
            y_pred_cls = (y_pred_classes == cls)
            
            # Per-class statistics
            TP = np.sum((y_true_cls) & (y_pred_cls))
            FP = np.sum((~y_true_cls) & (y_pred_cls))
            FN = np.sum((y_true_cls) & (~y_pred_cls))
            TN = np.sum((~y_true_cls) & (~y_pred_cls))
            
            # Compute metrics, handling division by zero
            cls_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            cls_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            cls_f1 = 2 * (cls_precision * cls_recall) / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            cls_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            
            # Store metrics
            per_class_metrics[int(cls)] = {
                'precision': float(cls_precision),
                'recall': float(cls_recall),
                'f1_score': float(cls_f1),
                'specificity': float(cls_specificity),
                'support': int(np.sum(y_true_cls))
            }
        
        # Save all metrics to file
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'f1_score_weighted': float(f1_weighted),
                'f1_score_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted)
            },
            'per_class': per_class_metrics
        }
        
        with open(os.path.join(self.model_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save prediction probabilities for further analysis
        np.save(os.path.join(self.model_path, 'y_pred_proba.npy'), y_pred_proba)
        np.save(os.path.join(self.model_path, 'y_pred_classes.npy'), y_pred_classes)
        np.save(os.path.join(self.model_path, 'y_test.npy'), y_test)
            
        return report, cm, y_pred_proba, metrics
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Plot accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.model_path, 'accuracy_history.png'))
        print(f"Accuracy history saved to {os.path.join(self.model_path, 'accuracy_history.png')}")
        plt.close()
        
        # Plot loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.model_path, 'loss_history.png'))
        print(f"Loss history saved to {os.path.join(self.model_path, 'loss_history.png')}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names=None):
        """Plot both regular and normalized confusion matrices"""
        os.makedirs(self.model_path, exist_ok=True)
        
        if class_names is None:
            # Default class names if not provided
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # Plot standard confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {os.path.join(self.model_path, 'confusion_matrix.png')}")
        plt.close()
        
        # Plot normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Replace NaN with 0
        cm_norm = np.nan_to_num(cm_norm)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'confusion_matrix_normalized.png'))
        print(f"Normalized confusion matrix saved to {os.path.join(self.model_path, 'confusion_matrix_normalized.png')}")
        plt.close()
    
    def plot_class_distribution(self, y_train, y_test, class_names=None):
        """Plot the distribution of classes in training and test sets"""
        os.makedirs(self.model_path, exist_ok=True)
        
        if class_names is None:
            # Default class names if not provided
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            
        plt.figure(figsize=(14, 6))
        
        # Training set distribution
        plt.subplot(1, 2, 1)
        train_counts = np.bincount(y_train, minlength=self.num_classes)
        sns.barplot(x=np.arange(len(class_names)), y=train_counts)
        plt.title('Class Distribution in Training Set')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
        
        # Test set distribution
        plt.subplot(1, 2, 2)
        test_counts = np.bincount(y_test, minlength=self.num_classes)
        sns.barplot(x=np.arange(len(class_names)), y=test_counts)
        plt.title('Class Distribution in Test Set')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'class_distribution.png'))
        print(f"Class distribution plot saved to {os.path.join(self.model_path, 'class_distribution.png')}")
        plt.close()
    
    def plot_roc_curves(self, y_test, y_pred_proba, class_names=None):
        """Plot ROC curves for each class"""
        os.makedirs(self.model_path, exist_ok=True)
        
        if class_names is None:
            # Default class names if not provided
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # One-hot encode the labels for ROC curve calculation
        y_test_binary = np.zeros((len(y_test), self.num_classes))
        for i, label in enumerate(y_test):
            y_test_binary[i, label] = 1
        
        plt.figure(figsize=(12, 10))
        
        # Calculate ROC curve and ROC area for each class
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.model_path, 'roc_curves.png'))
        print(f"ROC curves saved to {os.path.join(self.model_path, 'roc_curves.png')}")
        plt.close()
        
    def plot_precision_recall_curves(self, y_test, y_pred_proba, class_names=None):
        """Plot precision-recall curves for each class"""
        os.makedirs(self.model_path, exist_ok=True)
        
        if class_names is None:
            # Default class names if not provided
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # One-hot encode the labels
        y_test_binary = np.zeros((len(y_test), self.num_classes))
        for i, label in enumerate(y_test):
            y_test_binary[i, label] = 1
        
        plt.figure(figsize=(12, 10))
        
        # Calculate precision-recall curve for each class
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_test_binary[:, i], y_pred_proba[:, i])
            avg_precision = np.mean(precision)
            plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Multi-class Classification')
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.model_path, 'precision_recall_curves.png'))
        print(f"Precision-Recall curves saved to {os.path.join(self.model_path, 'precision_recall_curves.png')}")
        plt.close()
    
    def save_model(self):
        """Save the trained model and its configuration"""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save the model
        self.model.save(os.path.join(self.model_path, 'model.h5'))
        
        # Save model configuration
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'model_path': self.model_path
        }
        with open(os.path.join(self.model_path, 'config.json'), 'w') as f:
            json.dump(config, f)
    
    @classmethod
    def load_model(cls, model_path):
        """Load a trained model"""
        # Load configuration
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(
            input_shape=config['input_shape'],
            num_classes=config['num_classes'],
            model_path=config['model_path']
        )
        
        # Load weights
        model.model.load_weights(os.path.join(model_path, 'model.h5'))
        
        return model 