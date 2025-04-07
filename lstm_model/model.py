import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * x, dim=1)  # (batch_size, hidden_dim)
        return attended

class HybridLSTMGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super().__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,  # *2 for bidirectional
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # GRU processing
        gru_out, _ = self.gru(lstm_out)
        
        # Apply attention
        attended = self.attention(gru_out)
        
        # Final classification
        output = self.fc(attended)
        return output

def create_model(input_dim=30, hidden_dim=128, num_layers=2, num_classes=12, dropout=0.2):
    """Create and initialize the hybrid model"""
    model = HybridLSTMGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Initialize weights
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_normal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    
    return model

class NuclearAccidentClassifier:
    def __init__(self, input_shape, num_classes, model_path='lstm_model/saved_model'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_path = model_path
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a hybrid LSTM-GRU model for nuclear accident classification"""
        model = models.Sequential([
            # First LSTM layer with dropout
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # GRU layer
            layers.GRU(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with early stopping and model checkpointing"""
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate classification report
        report = classification_report(y_test, y_pred_classes)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        return report, cm
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
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