# user_specific_pipeline.py - Personalized posture pipeline for individual users
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, pipeline, Input, Output, Dataset, Model

@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "pandas", "numpy"]
)
def preprocess_user_posture_data(
    bucket_name: str,
    user_id: str,
    preprocessed_frames: Output[Dataset]
):
    """Load and preprocess CSV files for a specific user"""
    from google.cloud import storage
    import pandas as pd
    import numpy as np
    import json
    import os
    
    print(f"👤 Processing calibration data for user: {user_id}")
    print(f"🔍 Loading data from bucket: {bucket_name}")
    
    # Connect to Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Find CSV files for this specific user
    user_prefix = f"users/{user_id}/posture_data/"
    blobs = list(bucket.list_blobs(prefix=user_prefix))
    csv_files = [blob.name for blob in blobs if blob.name.endswith('.csv')]
    
    print(f"📁 Found {len(csv_files)} calibration files for {user_id}:")
    for file in csv_files:
        print(f"   📄 {file}")
    
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found for user {user_id} in {user_prefix}")
    
    # Create mapping from filename to posture code
    filename_to_posture = {
        'upright.csv': 'SP1',
        'slouching.csv': 'SP2', 
        'leaning-left.csv': 'SP3',
        'leaning-right.csv': 'SP4',
        'leaning-back-new.csv': 'SP5',
        'upright-right-leg-crossed.csv': 'SP6',
        'upright-left-leg-crossed.csv': 'SP7',
        'leaning-forward-slouching.csv': 'SP8',
        'edge-sitting.csv': 'SP9',
        'left-ankle-resting-right-leg.csv': 'SP10',
        'right-ankle-resting-left-leg.csv': 'SP11',
        'lounge-new.csv': 'SP12',
        'leaning-back-sitting-edge-new.csv': 'SP13',
        'leaning-back-left-ankle-resting-new.csv': 'SP14',
        'leaning-back-right-ankle-resting.csv': 'SP15',
        'leaning-back-left-leg-crossed.csv': 'SP16',
        'leaning-back-right-leg-crossed.csv': 'SP17',
        'left-rotating-trunk.csv': 'SP18',
        'right-rotating-trunk.csv': 'SP19'
    }
    
    # Download and load all CSV files for this user
    raw_dataset = {}
    os.makedirs("./temp_csvs", exist_ok=True)
    
    for blob_name in csv_files:
        filename = os.path.basename(blob_name)
        
        # Skip if we don't recognize this file
        if filename not in filename_to_posture:
            print(f"   ⚠️  Skipping unknown file: {filename}")
            continue
            
        posture_code = filename_to_posture[filename]
        
        # Download file
        local_path = f"./temp_csvs/{filename}"
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        
        # Load with pandas (no header, just like your working pipeline)
        data = pd.read_csv(local_path, header=None)
        raw_dataset[posture_code] = data
        
        print(f"   ✅ Loaded {filename} as {posture_code}: {data.shape[0]} rows")
    
    print(f"🎯 Loaded {len(raw_dataset)} posture datasets for user {user_id}")
    
    # Your exact preprocessing function (same as working pipeline)
    def process_frame(frame_data):
        sensor_size = 32
        seat_data = frame_data.iloc[:sensor_size].values.tolist()
        backrest_data = frame_data.iloc[sensor_size+1:(sensor_size*2)+1].values.tolist()
        seat_data.reverse()
        backrest_data.reverse()
        return {
            "seat": np.array(seat_data, dtype=float),
            "backrest": np.array(backrest_data, dtype=float)
        }
    
    # Process frames (same as working pipeline)
    all_frames = []
    
    for key, data in raw_dataset.items():
        print(f"🔄 Processing frames for {key}...")
        frame_num = 0
        frame_data = []
        frames_for_this_posture = 0

        for index, row in data.iterrows():
            if row.isnull().all():
                continue

            if 'Frame' in str(row.iloc[0]):
                if frame_data:
                    try:
                        processed_data = process_frame(pd.DataFrame(frame_data))
                        all_frames.append({
                            "posture": key,
                            "backrest": processed_data["backrest"].tolist(),
                            "seat": processed_data["seat"].tolist(),
                            "user_id": user_id  # Add user ID to each frame
                        })
                        frames_for_this_posture += 1
                    except Exception as e:
                        print(f"   ⚠️  Error processing frame {frame_num}: {str(e)}")
                    frame_data = []
                frame_num += 1
            else:
                frame_data.append(row)

        # Process the last frame
        if frame_data:
            try:
                processed_data = process_frame(pd.DataFrame(frame_data))
                all_frames.append({
                    "posture": key,
                    "backrest": processed_data["backrest"].tolist(),
                    "seat": processed_data["seat"].tolist(),
                    "user_id": user_id
                })
                frames_for_this_posture += 1
            except Exception as e:
                print(f"   ⚠️  Error processing final frame: {str(e)}")
                
        print(f'   ✅ Processed {key} posture with {frames_for_this_posture} frames')
    
    print(f"🎉 Total frames processed for {user_id}: {len(all_frames)}")
    
    # Count frames per posture
    posture_counts = {}
    for frame in all_frames:
        posture = frame["posture"]
        posture_counts[posture] = posture_counts.get(posture, 0) + 1
    
    print("📊 Frames per posture:")
    for posture, count in posture_counts.items():
        print(f"   {posture}: {count} frames")
    
    # Save processed frames
    os.makedirs(preprocessed_frames.path, exist_ok=True)
    
    with open(f"{preprocessed_frames.path}/all_frames.json", "w") as f:
        json.dump(all_frames, f)
    
    # Save metadata with user info
    metadata = {
        "user_id": user_id,
        "total_frames": len(all_frames),
        "posture_counts": posture_counts,
        "num_postures": len(posture_counts),
        "sensor_size": 32,
        "posture_timestamp": pd.Timestamp.now().isoformat(),
        "frame_structure": {
            "seat": "32 pressure sensors",
            "backrest": "32 pressure sensors", 
            "posture": "User-specific posture classification"
        }
    }
    
    with open(f"{preprocessed_frames.path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ User-specific preprocessing complete!")
    print(f"💾 Saved {len(all_frames)} processed frames for user {user_id}")

# Use the same augmentation component from your working pipeline
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scipy", "scikit-image"]
)
def augment_user_posture_data(
    preprocessed_frames: Input[Dataset],
    augmented_frames: Output[Dataset],
    sample_size_per_posture: int = 5  # Smaller for individual users
):
    """Augment user-specific posture data (same as working pipeline)"""
    import numpy as np
    import json
    import os
    import random
    from scipy.ndimage import gaussian_filter, map_coordinates, rotate
    
    # Load user data
    with open(f"{preprocessed_frames.path}/all_frames.json", "r") as f:
        all_frames = json.load(f)
    
    with open(f"{preprocessed_frames.path}/metadata.json", "r") as f:
        metadata = json.load(f)
    
    user_id = metadata["user_id"]
    print(f"👤 Augmenting data for user: {user_id}")
    print(f"🔄 Creating {sample_size_per_posture} augmented samples per frame...")
    print(f"📊 Original dataset: {len(all_frames)} frames")
    
    # Convert to numpy arrays
    for frame in all_frames:
        frame["backrest"] = np.array(frame["backrest"])
        frame["seat"] = np.array(frame["seat"])
    
    # Same augmentation functions as working pipeline
    def add_noise(data, noise_ratio=0.5):
        data_std = np.std(data)
        noise_level = noise_ratio * data_std
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def shift_data(data, shift_max=10):
        shift_amount = np.random.randint(-shift_max, shift_max + 1)
        shifted_data = np.roll(data, shift_amount)
        return np.clip(shifted_data, 0, 255)

    def rotate_data(data, angle):
        if len(data) == 1024:
            grid = data.reshape(32, 32)
            rotated_grid = rotate(grid, angle, reshape=False)
            return np.clip(rotated_grid.flatten(), 0, 255)
        elif len(data) == 32:
            return np.clip(data, 0, 255)
        else:
            return np.clip(data, 0, 255)

    def random_erasing(data, erase_prob=0.5, erase_size=0.1):
        if random.random() < erase_prob:
            data_copy = data.copy()
            length = len(data)
            erase_length = int(length * erase_size)
            start_idx = np.random.randint(0, length - erase_length)
            data_copy[start_idx:start_idx + erase_length] = 0
            return np.clip(data_copy, 0, 255)
        return np.clip(data, 0, 255)

    def elastic_transform(data, alpha, sigma):
        try:
            if len(data) == 1024:
                grid = data.reshape(32, 32)
                random_state = np.random.RandomState(None)
                shape = grid.shape
                dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
                dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
                x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
                indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
                transformed = map_coordinates(grid, indices, order=1, mode='reflect').reshape(shape)
                return np.clip(transformed.flatten(), 0, 255)
            elif len(data) == 32:
                return np.clip(data, 0, 255)
            else:
                return np.clip(data, 0, 255)
        except:
            return np.clip(data, 0, 255)

    def complex_augment_data(data, sample_size=10):
        augmented_data = []
        for _ in range(sample_size):
            aug_data = data.copy()
            if random.choice([True, False]):
                aug_data = shift_data(aug_data, shift_max=np.random.randint(1, 6))
            if random.choice([True, False]):
                aug_data = rotate_data(aug_data, angle=np.random.uniform(-30, 30))
            if random.choice([True, False]):
                aug_data = random_erasing(aug_data)
            if random.choice([True, False]):
                aug_data = elastic_transform(aug_data, alpha=24, sigma=4)
            if random.choice([True, False]):
                aug_data = add_noise(aug_data)
            augmented_data.append(aug_data)
        return augmented_data

    # Generate augmented dataset for this user
    augmented_dataset = []
    
    for i, frame in enumerate(all_frames):
        if i % 10 == 0:
            print(f"   Processing frame {i+1}/{len(all_frames)}")
            
        posture_name = frame["posture"]
        
        augmented_backrest = complex_augment_data(frame["backrest"], sample_size=sample_size_per_posture)
        augmented_seat = complex_augment_data(frame["seat"], sample_size=sample_size_per_posture)

        for backrest, seat in zip(augmented_backrest, augmented_seat):
            augmented_dataset.append({
                "posture": posture_name, 
                "backrest": backrest.tolist(), 
                "seat": seat.tolist(),
                "user_id": user_id
            })
    
    print(f"✅ Augmentation complete for user {user_id}!")
    print(f"📊 Original dataset size: {len(all_frames)}")
    print(f"📊 Augmented dataset size: {len(augmented_dataset)}")
    
    # Convert numpy arrays back to lists for JSON storage
    for frame in all_frames:
        frame["backrest"] = frame["backrest"].tolist()
        frame["seat"] = frame["seat"].tolist()
    
    # Save augmented data
    os.makedirs(augmented_frames.path, exist_ok=True)
    
    with open(f"{augmented_frames.path}/original_frames.json", "w") as f:
        json.dump(all_frames, f)
    
    with open(f"{augmented_frames.path}/augmented_frames.json", "w") as f:
        json.dump(augmented_dataset, f)
    
    # Save metadata
    augmentation_metadata = {
        **metadata,
        "original_frame_count": len(all_frames),
        "augmented_frame_count": len(augmented_dataset),
        "augmentation_ratio": sample_size_per_posture,
        "total_frames_after_augmentation": len(all_frames) + len(augmented_dataset)
    }
    
    with open(f"{augmented_frames.path}/augmentation_metadata.json", "w") as f:
        json.dump(augmentation_metadata, f, indent=2)
    
    print(f"💾 Saved augmented data for user {user_id}")

# Include the remaining pipeline components from your working pipeline
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy"]
)
def normalize_user_posture_data(
    augmented_frames: Input[Dataset],
    normalized_frames: Output[Dataset]
):
    """Normalize the user's data using your exact Colab normalization"""
    import numpy as np
    import json
    import os
    
    print("🔄 Starting data normalization for user...")
    
    # Load augmented data
    with open(f"{augmented_frames.path}/original_frames.json", "r") as f:
        all_frames = json.load(f)
    
    with open(f"{augmented_frames.path}/augmented_frames.json", "r") as f:
        augmented_dataset = json.load(f)
    
    with open(f"{augmented_frames.path}/augmentation_metadata.json", "r") as f:
        metadata = json.load(f)
    
    user_id = metadata["user_id"]
    print(f"👤 Normalizing data for user: {user_id}")
    print(f"📊 Original frames: {len(all_frames)}")
    print(f"📊 Augmented frames: {len(augmented_dataset)}")
    
    # Your exact sensor data extraction function
    def extract_sensor_data(dataset):
        backrest_data = []
        seat_data = []
        for frame in dataset:
            backrest_data.append(frame["backrest"])
            seat_data.append(frame["seat"])
        return np.array(backrest_data), np.array(seat_data)

    # Extract data from both datasets
    orig_backrest, orig_seat = extract_sensor_data(all_frames)
    aug_backrest, aug_seat = extract_sensor_data(augmented_dataset)

    # Combine all sensor data to find global statistics
    all_backrest = np.concatenate([orig_backrest, aug_backrest])
    all_seat = np.concatenate([orig_seat, aug_seat])

    # Calculate global min/max for each sensor type
    backrest_min = all_backrest.min()
    backrest_max = all_backrest.max()
    seat_min = all_seat.min()
    seat_max = all_seat.max()

    print(f"📏 Backrest range: [{backrest_min:.2f}, {backrest_max:.2f}]")
    print(f"📏 Seat range: [{seat_min:.2f}, {seat_max:.2f}]")

    # Your exact normalize function
    def normalize_dataset(dataset, backrest_min, backrest_max, seat_min, seat_max):
        normalized = []
        for frame in dataset:
            backrest_array = np.array(frame["backrest"])
            seat_array = np.array(frame["seat"])
            
            normalized_frame = {
                "posture": frame["posture"],
                "backrest": ((backrest_array - backrest_min) / (backrest_max - backrest_min)).tolist(),
                "seat": ((seat_array - seat_min) / (seat_max - seat_min)).tolist(),
                "user_id": frame.get("user_id", metadata["user_id"])
            }
            normalized.append(normalized_frame)
        return normalized

    # Normalize both datasets using global statistics
    print("⚖️ Normalizing original dataset...")
    normalized_original = normalize_dataset(all_frames, backrest_min, backrest_max, seat_min, seat_max)
    
    print("⚖️ Normalizing augmented dataset...")
    normalized_augmented = normalize_dataset(augmented_dataset, backrest_min, backrest_max, seat_min, seat_max)

    print("✅ Normalization completed!")
    print(f"📊 Normalized original dataset size: {len(normalized_original)}")
    print(f"📊 Normalized augmented dataset size: {len(normalized_augmented)}")
    
    # Save normalized data
    os.makedirs(normalized_frames.path, exist_ok=True)
    
    with open(f"{normalized_frames.path}/normalized_original.json", "w") as f:
        json.dump(normalized_original, f)
    
    with open(f"{normalized_frames.path}/normalized_augmented.json", "w") as f:
        json.dump(normalized_augmented, f)
    
    # Save normalization parameters and metadata
    normalization_metadata = {
        **metadata,
        "normalization_params": {
            "backrest_min": float(backrest_min),
            "backrest_max": float(backrest_max),
            "seat_min": float(seat_min),
            "seat_max": float(seat_max)
        },
        "normalized_original_count": len(normalized_original),
        "normalized_augmented_count": len(normalized_augmented),
        "total_normalized_frames": len(normalized_original) + len(normalized_augmented)
    }
    
    with open(f"{normalized_frames.path}/normalization_metadata.json", "w") as f:
        json.dump(normalization_metadata, f, indent=2)
    
    print(f"💾 Saved normalized data for user {user_id}")

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def flatten_and_split_user_data(
    normalized_frames: Input[Dataset],
    train_val_test_data: Output[Dataset]
):
    """Flatten and split user data exactly like your working pipeline"""
    import numpy as np
    import json
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    print("🔄 Flattening matrices and creating train/val/test splits for user...")
    
    # Load normalized data
    with open(f"{normalized_frames.path}/normalized_original.json", "r") as f:
        normalized_original = json.load(f)
    
    with open(f"{normalized_frames.path}/normalized_augmented.json", "r") as f:
        normalized_augmented = json.load(f)
    
    with open(f"{normalized_frames.path}/normalization_metadata.json", "r") as f:
        metadata = json.load(f)
    
    user_id = metadata["user_id"]
    print(f"👤 Processing data for user: {user_id}")
    print(f"📊 Original normalized data: {len(normalized_original)} samples")
    print(f"📊 Augmented normalized data: {len(normalized_augmented)} samples")
    
    # Your exact flatten function
    def flatten_matrix(matrix):
        return np.array(matrix).flatten()

    # Prepare the data for training - your exact logic
    features = []
    labels = []

    print("🔧 Flattening backrest and seat data...")
    for data_point in np.concatenate((normalized_original, normalized_augmented)):
        flat_backrest = flatten_matrix(data_point['backrest'])
        flat_seat = flatten_matrix(data_point['seat'])
        features.append(np.concatenate((flat_backrest, flat_seat)))
        labels.append(data_point['posture'])

    X = np.array(features)
    y = np.array(labels)
    
    print(f"📏 Feature matrix shape: {X.shape}")
    print(f"🏷️ Labels shape: {y.shape}")
    print(f"🎯 Features per sample: {X.shape[1]} (32 backrest + 32 seat = 64 total)")
    
    # Check if we have enough data for splitting
    if len(X) < 10:
        raise ValueError(f"Not enough data for user {user_id}. Need at least 10 samples, got {len(X)}")
    
    # Encode the labels - your exact approach
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"🔢 Unique labels: {len(np.unique(y_encoded))}")
    print(f"🏷️ Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Split the data - adjust for smaller datasets
    # For individual users, we might have less data, so adjust split ratios
    if len(X) < 50:
        # Smaller dataset - use simpler split
        test_size = min(0.3, max(0.1, 5/len(X)))  # At least 1 sample, max 30%
        val_size = min(0.2, max(0.1, 3/len(X)))   # At least 1 sample, max 20%
    else:
        # Normal split ratios
        test_size = 0.2
        val_size = 0.2
    
    try:
        # First split: separate test set
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Second split: separate validation from training
        if len(X_train_temp) > 3:  # Only create validation set if we have enough data
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_temp, y_train_temp, test_size=val_size, random_state=42, stratify=y_train_temp
            )
        else:
            # Too little data for validation set
            X_train, X_val = X_train_temp, X_train_temp
            y_train, y_val = y_train_temp, y_train_temp
    
    except ValueError as e:
        # If stratified split fails due to small classes, use regular split
        print(f"⚠️ Stratified split failed, using regular split: {str(e)}")
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        if len(X_train_temp) > 3:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_temp, y_train_temp, test_size=val_size, random_state=42
            )
        else:
            X_train, X_val = X_train_temp, X_train_temp
            y_train, y_val = y_train_temp, y_train_temp

    print(f"📈 Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"📊 Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"📉 Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Save split data
    os.makedirs(train_val_test_data.path, exist_ok=True)
    
    # Save as numpy arrays
    np.save(f"{train_val_test_data.path}/X_train.npy", X_train)
    np.save(f"{train_val_test_data.path}/X_val.npy", X_val)
    np.save(f"{train_val_test_data.path}/X_test.npy", X_test)
    np.save(f"{train_val_test_data.path}/y_train.npy", y_train)
    np.save(f"{train_val_test_data.path}/y_val.npy", y_val)
    np.save(f"{train_val_test_data.path}/y_test.npy", y_test)
    
    # Save label encoder and metadata
    import joblib
    joblib.dump(label_encoder, f"{train_val_test_data.path}/label_encoder.pkl")
    
    split_metadata = {
        **metadata,
        "total_samples": len(X),
        "feature_dimension": X.shape[1],
        "num_classes": len(np.unique(y_encoded)),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_percentage": len(X_train)/len(X)*100,
        "val_percentage": len(X_val)/len(X)*100,
        "test_percentage": len(X_test)/len(X)*100,
        "label_classes": label_encoder.classes_.tolist(),
        "data_shape_per_sample": {
            "total_features": X.shape[1],
            "backrest_features": X.shape[1] // 2,
            "seat_features": X.shape[1] // 2
        }
    }
    
    with open(f"{train_val_test_data.path}/split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)
    
    print("✅ Data flattening and splitting complete!")
    print(f"💾 Saved train/val/test data for user {user_id}")

@component(
    base_image="gcr.io/deeplearning-platform-release/tf2-gpu.2-11:latest",
    packages_to_install=["scikit-learn", "joblib"]
)
def train_user_cnn_model(
    train_val_test_data: Input[Dataset],
    cnn_model: Output[Model],
    epochs: int = 30,
    batch_size: int = 8
):
    """Train user-specific CNN model"""
    import numpy as np
    import json
    import os
    import joblib
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import classification_report, accuracy_score
    
    # Load split data
    X_train = np.load(f"{train_val_test_data.path}/X_train.npy")
    X_val = np.load(f"{train_val_test_data.path}/X_val.npy")
    X_test = np.load(f"{train_val_test_data.path}/X_test.npy")
    y_train = np.load(f"{train_val_test_data.path}/y_train.npy")
    y_val = np.load(f"{train_val_test_data.path}/y_val.npy")
    y_test = np.load(f"{train_val_test_data.path}/y_test.npy")
    
    label_encoder = joblib.load(f"{train_val_test_data.path}/label_encoder.pkl")
    
    with open(f"{train_val_test_data.path}/split_metadata.json", "r") as f:
        metadata = json.load(f)
    
    user_id = metadata["user_id"]
    num_classes = metadata["num_classes"]
    
    print(f"🎯 Training personalized CNN for user: {user_id}")
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Validation data shape: {X_val.shape}")
    print(f"📊 Test data shape: {X_test.shape}")
    print(f"🏷️ Number of classes: {num_classes}")
    print(f"🏷️ Classes: {metadata['label_classes']}")
    
    # Reshape data for CNN 
    def reshape_for_cnn(X):
        # X shape: (samples, total_features) where total_features = backrest + seat
        samples = X.shape[0]
        total_features = X.shape[1]
        features_per_sensor = total_features // 2  # Half for backrest, half for seat
        
        # Split back into backrest and seat
        backrest = X[:, :features_per_sensor]  # First half
        seat = X[:, features_per_sensor:]      # Second half
        
        if features_per_sensor == 1024:  # 32x32 sensors each
            # Already the right shape for 32x32
            backrest_grid = backrest.reshape(samples, 32, 32)
            seat_grid = seat.reshape(samples, 32, 32)
            
            # Stack as 2 channels (32, 32, 2)
            combined = np.stack([backrest_grid, seat_grid], axis=-1)
            
        elif features_per_sensor == 32:  # 32 sensors each (need to pad to 32x32)
            # Reshape to some 2D grid and pad to 32x32
            # Let's assume 4x8 for now, but this should be based on your actual sensor layout
            backrest_grid = backrest.reshape(samples, 4, 8)
            seat_grid = seat.reshape(samples, 4, 8)
            
            # Pad from 4x8 to 32x32
            backrest_padded = np.pad(backrest_grid, ((0, 0), (14, 14), (12, 12)), mode='constant')
            seat_padded = np.pad(seat_grid, ((0, 0), (14, 14), (12, 12)), mode='constant')
            
            # Stack as 2 channels
            combined = np.stack([backrest_padded, seat_padded], axis=-1)
            
        else:
            raise ValueError(f"Unexpected number of features per sensor: {features_per_sensor}")
        
        return combined
    
    print("🔄 Reshaping data for CNN...")
    X_train_upscaled = reshape_for_cnn(X_train)
    X_val_upscaled = reshape_for_cnn(X_val)
    X_test_upscaled = reshape_for_cnn(X_test)
    
    print(f"📐 Reshaped training data: {X_train_upscaled.shape}")
    
    # Convert labels to categorical
    y_train_categorical = to_categorical(y_train, num_classes)
    y_val_categorical = to_categorical(y_val, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)
    
    # Create CNN model (same architecture as your working pipeline)
    def create_cnn_model(input_shape, num_classes):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    input_shape = (32, 32, 2)
    model = create_cnn_model(input_shape, num_classes)
    
    print("🏗️ User-specific CNN Model Architecture:")
    model.summary()
    
    # Early stopping (same as working pipeline)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print(f"🚀 Starting personalized CNN training for user {user_id}...")
    history = model.fit(
        X_train_upscaled, y_train_categorical,
        validation_data=(X_val_upscaled, y_val_categorical),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    print(f"📊 Evaluating personalized model for user {user_id}...")
    
    train_loss, train_accuracy = model.evaluate(X_train_upscaled, y_train_categorical, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val_upscaled, y_val_categorical, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test_upscaled, y_test_categorical, verbose=0)
    
    print(f"🎯 User {user_id} - Training Accuracy: {train_accuracy:.4f}")
    print(f"🎯 User {user_id} - Validation Accuracy: {val_accuracy:.4f}")
    print(f"🎯 User {user_id} - Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed classification report
    y_test_pred = model.predict(X_test_upscaled)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    
    print(f"\n📋 Test Set Classification Report for user {user_id}:")
    print(classification_report(y_test, y_test_pred_classes, target_names=metadata['label_classes']))
    
    # Save the personalized model
    os.makedirs(cnn_model.path, exist_ok=True)
    model.save(f"{cnn_model.path}/posture_cnn_model")
    
    # Save training history
    history_dict = {
        "loss": [float(x) for x in history.history['loss']],
        "accuracy": [float(x) for x in history.history['accuracy']],
        "val_loss": [float(x) for x in history.history['val_loss']],
        "val_accuracy": [float(x) for x in history.history['val_accuracy']]
    }
    
    with open(f"{cnn_model.path}/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    
    # Save model metadata
    cnn_metadata = {
        **metadata,
        "model_type": "Personalized_CNN",
        "input_shape": input_shape,
        "epochs_trained": len(history.history['loss']),
        "early_stopping_patience": 5,
        "final_train_accuracy": float(train_accuracy),
        "final_val_accuracy": float(val_accuracy),
        "final_test_accuracy": float(test_accuracy),
        "final_train_loss": float(train_loss),
        "final_val_loss": float(val_loss),
        "final_test_loss": float(test_loss),
        "batch_size": batch_size,
        "personalized_for_user": user_id,
        "architecture": {
            "conv1": "32 filters, 3x3, relu",
            "pool1": "2x2 maxpool",
            "conv2": "64 filters, 3x3, relu", 
            "pool2": "2x2 maxpool",
            "dense1": "128 units, relu",
            "output": f"{num_classes} units, softmax"
        }
    }
    
    with open(f"{cnn_model.path}/cnn_metadata.json", "w") as f:
        json.dump(cnn_metadata, f, indent=2)
    
    # Save label encoder for inference
    joblib.dump(label_encoder, f"{cnn_model.path}/label_encoder.pkl")
    
    print(f"✅ Personalized CNN training complete for user {user_id}!")
    print(f"💾 Model saved to: {cnn_model.path}")
    print(f"📈 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"⏹️ Training stopped after {len(history.history['loss'])} epochs")
    
@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def register_user_model(user_id: str, model: Input[Model]):
    from google.cloud import aiplatform
    import os

    aiplatform.init(project=os.getenv("GCLOUD_PROJECT_ID"), location="europe-west2")

    aiplatform.Model.upload(
        display_name=f"user_model_{user_id}",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        description=f"Personalized CNN model for user {user_id}"
    )


@component(
    base_image="python:3.9",
    packages_to_install=["requests"]
)
def trigger_webhook(user_id: str, webhook_url: str):
    """
    Triggers a webhook with the user ID as JSON payload.
    """
    # Create payload safely
    payload = json.dumps({"userId": user_id})

    # Set headers
    headers = {"Content-Type": "application/json"}

    # Send POST request
    response = requests.post(webhook_url, data=payload, headers=headers)

    print(f"Webhook response status: {response.status_code}")
    print(f"Webhook response body: {response.text}")

    # Optionally raise exception if webhook failed
    response.raise_for_status()

@pipeline(
    name="user-specific-posture-pipeline",
    description="Personalized posture classification pipeline for individual users"
)
def user_specific_posture_pipeline(
    bucket_name: str,
    user_id: str,
    augmentation_samples: int = 5,
    cnn_epochs: int = 30,
    cnn_batch_size: int = 8
):
    """Personalized pipeline for training user-specific posture models"""
    
    # Step 1: Preprocess user's calibration data
    preprocess_task = preprocess_user_posture_data(
        bucket_name=bucket_name,
        user_id=user_id
    )
    
    # Step 2: Augment the user's data
    augment_task = augment_user_posture_data(
        preprocessed_frames=preprocess_task.outputs["preprocessed_frames"],
        sample_size_per_posture=augmentation_samples
    )
    
    # Step 3: Normalize the data
    normalize_task = normalize_user_posture_data(
        augmented_frames=augment_task.outputs["augmented_frames"]
    )
    
    # Step 4: Flatten and create train/val/test splits
    split_task = flatten_and_split_user_data(
        normalized_frames=normalize_task.outputs["normalized_frames"]
    )
    
    # Step 5: Train user-specific CNN
    cnn_task = train_user_cnn_model(
        train_val_test_data=split_task.outputs["train_val_test_data"],
        epochs=cnn_epochs,
        batch_size=cnn_batch_size
    )
    
    # Step 6: Deploy model to Registry
    register_model_task = register_user_model(
        user_id=user_id,
        model=cnn_task.outputs["cnn_model"]
    )
    
    # Step 7: Trigger Webhook
    # webhook_task = trigger_webhook(
    #     user_id=user_id,
    #     webhook_url="https://pipelinewebhook-hfarmdvsyq-uc.a.run.app"
    # )
    

# Compile the user-specific pipeline
if __name__ == "__main__":
    print("🔧 Compiling user-specific posture pipeline...")
    
    compiler.Compiler().compile(
        pipeline_func=user_specific_posture_pipeline,
        package_path="user_specific_posture_pipeline.json"
    )
    
    print("✅ User-specific pipeline compiled!")
    print("📁 Created file: user_specific_posture_pipeline.json")
    print("\n👤 This pipeline creates personalized models for individual users")
    print("📊 Expected folder structure:")
    print("  gs://bucket/users/{user_id}/posture_data/")
    print("    ├── upright.csv")
    print("    ├── slouching.csv")
    print("    ├── leaning-left.csv")
    print("    └── ...")
    print("\n🎯 Usage:")
    print("  - user_id: 'user_123' (unique identifier)")
    print("  - bucket_name: your storage bucket")
    print("  - Model saved to: users/{user_id}/models/")
