import numpy as np
import os
import json
from model import NuclearAccidentClassifier

def load_preprocessed_data():
    """Load preprocessed data from the processed_data directory"""
    data_dir = '../processed_data'
    
    # Load training data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    # Load validation data
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Load preprocessing info
    with open(os.path.join(data_dir, 'preprocessing_info.json'), 'r') as f:
        preprocessing_info = json.load(f)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessing_info

def main():
    # Load preprocessed data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessing_info = load_preprocessed_data()
    
    # Get input shape and number of classes
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
    num_classes = len(np.unique(y_train))
    
    # Create and train model
    model = NuclearAccidentClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        model_path='lstm_model/saved_model'
    )
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )
    
    # Plot training history
    model.plot_training_history(history)
    
    # Evaluate the model
    report, cm = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    model.plot_confusion_matrix(cm)
    
    # Save the model
    model.save_model()
    
    # Print classification report
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open('lstm_model/saved_model/classification_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main() 