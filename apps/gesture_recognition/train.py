"""
Hand Gesture Recognition using MediaPipe Hand Landmarks

This implementation is based on and adapted from:
    "hand-gesture-recognition-using-mediapipe" by Kazuhito Takahashi (Kazuhito00)
    Original repository: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    Licensed under Apache License 2.0
"""

import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


GESTURES = ["thumbs_up", "peace_sign", "closed_fist", "open_palm"]
NUM_CLASSES = len(GESTURES)

def load_dataset(csv_path='./apps/gesture_recognition/csv/keypoint.csv'):
    """Load dataset from CSV file"""
    X = []
    y = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) > 1:  # Make sure we have data
                    label = int(row[0])
                    landmarks = [float(x) for x in row[1:]]
                    X.append(landmarks)
                    y.append(label)
    except FileNotFoundError:
        print(f"Dataset file {csv_path} not found!")
        print("Please run collect_keypoints.py first to collect training data.")
        return None, None
    
    if not X:
        print("No training data found in the CSV file!")
        return None, None
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def create_model(input_shape, num_classes):
    """Create a simple MLP model for gesture classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, gesture_labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=gesture_labels, yticklabels=gesture_labels)
    plt.title('Gesture Recognition Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./apps/gesture_recognition/csv/confusion_matrix.png')
    plt.show()

def main():
    print("Loading dataset...")
    X, y = load_dataset()
    
    if X is None or y is None:
        return
    
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train model
    print("Creating model...")
    model = create_model(X.shape[1], NUM_CLASSES)
    print(model.summary())
    
    # Training callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=0.001
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=1000,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=GESTURES))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, GESTURES)
    
    # Convert to TensorFlow Lite
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_model_path = './models/gesture_classifier.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model saved as {tflite_model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./apps/gesture_recognition/csv/training_history.png')
    plt.show()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
