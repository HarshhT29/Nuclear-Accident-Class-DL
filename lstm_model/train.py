import os
import numpy as np
import json
from model import NuclearAccidentClassifier

def load_preprocessed_data():
    """Load preprocessed data from the processed_data directory"""
    data_dir = 'processed_data'
    
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
    
    # Get accident type names from preprocessing info
    accident_types = list(preprocessing_info['label_mapping'].values())
    if not accident_types:
        accident_types = ['LOCA', 'SGBTR', 'LR', 'MD', 'SGATR', 'SLBIC', 
                         'LOCAC', 'RI', 'FLB', 'LLB', 'SLBOC', 'RW']
    
    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Create output directory
    model_dir = 'lstm_model/saved_model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and train model
    model = NuclearAccidentClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        model_path=model_dir
    )
    
    # Plot class distribution before training
    model.plot_class_distribution(y_train, y_test, accident_types)
    
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
    report, cm, y_pred_proba, metrics = model.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    model.plot_confusion_matrix(cm, accident_types)
    
    # Plot ROC curves
    model.plot_roc_curves(y_test, y_pred_proba, accident_types)
    
    # Plot precision-recall curves
    model.plot_precision_recall_curves(y_test, y_pred_proba, accident_types)
    
    # Save the model
    model.save_model()
    
    # Print classification report
    print("\nClassification Report:")
    print(report)
    
    # Print metrics summary
    print("\nPerformance Metrics Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Save classification report
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print(f"\nAll results saved to {model_dir}")
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 