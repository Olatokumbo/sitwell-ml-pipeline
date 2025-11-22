from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Model, Dataset

@component(
    base_image="gcr.io/deeplearning-platform-release/tf2-gpu.2-11:latest",
    packages_to_install=["scikit-learn"]
)
def train_gtrace_cnn(
    gtrace_train: Input[Dataset],
    gtrace_val: Input[Dataset],
    gtrace_test: Input[Dataset],
    gtrace_cnn_model: Output[Model],
    epochs: int = 30,
    batch_size: int = 8
):
    """Train CNN model for GTrace sensor data (gtrace_back + gtrace_seat)"""
    import numpy as np
    import json
    import os
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"{'='*60}")
    print(f"GTRACE CNN TRAINING")
    print(f"{'='*60}")
    
    # Load train data
    with open(f"{gtrace_train.path}/gtrace_train_frames.json", "r") as f:
        train_frames = json.load(f)
    with open(f"{gtrace_train.path}/gtrace_train_metadata.json", "r") as f:
        train_metadata = json.load(f)
    
    # Load validation data
    with open(f"{gtrace_val.path}/gtrace_val_frames.json", "r") as f:
        val_frames = json.load(f)
    
    # Load test data
    with open(f"{gtrace_test.path}/gtrace_test_frames.json", "r") as f:
        test_frames = json.load(f)
    
    user_id = train_metadata["user_id"]
    print(f"👤 Training GTrace CNN for user: {user_id}")
    print(f"📊 Train samples: {len(train_frames)}")
    print(f"📊 Validation samples: {len(val_frames)}")
    print(f"📊 Test samples: {len(test_frames)}")
    
    # Create label mapping
    posture_classes = sorted(list(set([f["posture"] for f in train_frames])))
    num_classes = len(posture_classes)
    posture_to_idx = {posture: idx for idx, posture in enumerate(posture_classes)}
    idx_to_posture = {idx: posture for posture, idx in posture_to_idx.items()}
    
    print(f"🏷️ Number of classes: {num_classes}")
    print(f"🏷️ Classes: {posture_classes}")
    
    # Prepare data for CNN
    def prepare_gtrace_data(frames, posture_to_idx):
        """Convert frames to CNN input format"""
        X = []
        y = []
        
        for frame in frames:
            # Stack gtrace_back and gtrace_seat as 2 channels
            gtrace_back = np.array(frame["gtrace_back"])
            gtrace_seat = np.array(frame["gtrace_seat"])
            
            # Stack as (height, width, channels)
            combined = np.stack([gtrace_back, gtrace_seat], axis=-1)
            X.append(combined)
            
            # Get label
            y.append(posture_to_idx[frame["posture"]])
        
        return np.array(X), np.array(y)
    
    print("\n🔄 Preparing GTrace data for CNN...")
    X_train, y_train = prepare_gtrace_data(train_frames, posture_to_idx)
    X_val, y_val = prepare_gtrace_data(val_frames, posture_to_idx)
    X_test, y_test = prepare_gtrace_data(test_frames, posture_to_idx)
    
    print(f"📐 X_train shape: {X_train.shape}")
    print(f"📐 X_val shape: {X_val.shape}")
    print(f"📐 X_test shape: {X_test.shape}")
    print(f"📐 Input shape: {X_train.shape[1:]}")
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    
    # Build GTrace CNN model
    def create_gtrace_cnn(input_shape, num_classes):
        """
        CNN architecture for GTrace data
        Input: (height, width, 2) where 2 channels are gtrace_back and gtrace_seat
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    input_shape = X_train.shape[1:]  # (height, width, 2)
    model = create_gtrace_cnn(input_shape, num_classes)
    
    print("\n🏗️ GTrace CNN Model Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print(f"\n🚀 Training GTrace CNN for user {user_id}...")
    print(f"⚙️ Epochs: {epochs}, Batch size: {batch_size}")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print(f"\n📊 Evaluating GTrace model for user {user_id}...")
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    print(f"\n🎯 GTrace CNN Results:")
    print(f"   Training Accuracy: {train_acc:.4f} (Loss: {train_loss:.4f})")
    print(f"   Validation Accuracy: {val_acc:.4f} (Loss: {val_loss:.4f})")
    print(f"   Test Accuracy: {test_acc:.4f} (Loss: {test_loss:.4f})")
    
    # Generate predictions
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    
    # Classification report
    print(f"\n📋 GTrace Test Classification Report:")
    print(classification_report(
        y_test, 
        y_test_pred_classes, 
        target_names=posture_classes,
        digits=4
    ))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred_classes)
    print(f"\n📊 Confusion Matrix:")
    print(conf_matrix)
    
    # Save model
    os.makedirs(gtrace_cnn_model.path, exist_ok=True)
    # Save in SavedModel format for Vertex AI
    model.save(
        f"{gtrace_cnn_model.path}",  # No .h5 extension!
        save_format='tf'  # Explicitly use TensorFlow SavedModel format
    )
    
    print(f"✅ Model saved in SavedModel format to: {gtrace_cnn_model.path}")
    
    # Verify the saved model structure
    saved_model_path = f"{gtrace_cnn_model.path}"
    if os.path.exists(f"{saved_model_path}/saved_model.pb"):
        print(f"✅ Verified: saved_model.pb exists")
    else:
        print(f"⚠️  Warning: saved_model.pb not found!")
    
    # List contents for debugging
    print(f"\n📁 Contents of {saved_model_path}:")
    for root, dirs, files in os.walk(saved_model_path):
        level = root.replace(saved_model_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    
    # Save metadata
    gtrace_model_metadata = {
        "user_id": user_id,
        "model_type": "GTrace_CNN",
        "modality": "gtrace",
        "sensor_channels": ["gtrace_back", "gtrace_seat"],
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "posture_classes": posture_classes,
        "posture_to_idx": posture_to_idx,
        "idx_to_posture": idx_to_posture,
        "model_format": "tensorflow_savedmodel",
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "epochs_trained": len(history.history['loss'])
        },
        "performance": {
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "best_val_accuracy": float(max(history.history['val_accuracy']))
        },
        "data_split": {
            "train_samples": len(train_frames),
            "val_samples": len(val_frames),
            "test_samples": len(test_frames)
        }
    }
    
    with open(f"{gtrace_cnn_model.path}/gtrace_model_metadata.json", "w") as f:
        json.dump(gtrace_model_metadata, f, indent=2)
    
    # Save training history
    history_dict = {
        "loss": [float(x) for x in history.history['loss']],
        "accuracy": [float(x) for x in history.history['accuracy']],
        "val_loss": [float(x) for x in history.history['val_loss']],
        "val_accuracy": [float(x) for x in history.history['val_accuracy']]
    }
    
    with open(f"{gtrace_cnn_model.path}/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Save confusion matrix
    conf_matrix_dict = {
        "confusion_matrix": conf_matrix.tolist(),
        "labels": posture_classes
    }
    
    with open(f"{gtrace_cnn_model.path}/confusion_matrix.json", "w") as f:
        json.dump(conf_matrix_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ GTRACE CNN TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"💾 Model saved to: {gtrace_cnn_model.path}")
    print(f"🎯 Final test accuracy: {test_acc:.4f}")