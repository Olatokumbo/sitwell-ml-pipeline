from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def normalize_confmat_and_gtrace_data(
    augmented_frames: Input[Dataset],
    normalized_confmat: Output[Dataset],
    normalized_gtrace: Output[Dataset],
    confmat_scaler: Output[Model],
    gtrace_scaler: Output[Model]
):
    """Normalize both ConfMat and GTrace data separately for dual CNN training"""
    import numpy as np
    import json
    import os
    import pickle
    from sklearn.preprocessing import StandardScaler
    
    # Load augmented data
    with open(f"{augmented_frames.path}/original_frames.json", "r") as f:
        original_frames = json.load(f)
    
    with open(f"{augmented_frames.path}/augmented_frames.json", "r") as f:
        augmented_dataset = json.load(f)
    
    with open(f"{augmented_frames.path}/augmentation_metadata.json", "r") as f:
        metadata = json.load(f)
    
    user_id = metadata["user_id"]
    print(f"👤 Normalizing data for user: {user_id}")
    print("🔹 Processing both ConfMat and GTrace modalities")
    print(f"📊 Original frames: {len(original_frames)}")
    print(f"📊 Augmented frames: {len(augmented_dataset)}")
    
    # Combine original and augmented frames
    all_frames = original_frames + augmented_dataset
    print(f"📊 Total frames to normalize: {len(all_frames)}")
    
    # Get original dimensions from first frame
    sample_frame = all_frames[0]
    conf_back_shape = np.array(sample_frame["conf_back"]).shape
    conf_seat_shape = np.array(sample_frame["conf_seat"]).shape
    gtrace_back_shape = np.array(sample_frame["gtrace_back"]).shape
    gtrace_seat_shape = np.array(sample_frame["gtrace_seat"]).shape
    
    conf_back_size = conf_back_shape[0] * conf_back_shape[1]
    conf_seat_size = conf_seat_shape[0] * conf_seat_shape[1]
    gtrace_back_size = gtrace_back_shape[0] * gtrace_back_shape[1]
    gtrace_seat_size = gtrace_seat_shape[0] * gtrace_seat_shape[1]
    
    print(f"\n📐 Channel dimensions:")
    print(f"   ConfMat: conf_back {conf_back_shape}, conf_seat {conf_seat_shape}")
    print(f"   GTrace: gtrace_back {gtrace_back_shape}, gtrace_seat {gtrace_seat_shape}")
    
    # ========================================
    # NORMALIZE CONFMAT DATA
    # ========================================
    print(f"\n{'='*60}")
    print(f"🔹 NORMALIZING CONFMAT DATA")
    print(f"{'='*60}")
    
    all_confmat_data = []
    for frame in all_frames:
        conf_back_flat = np.array(frame["conf_back"]).flatten()
        conf_seat_flat = np.array(frame["conf_seat"]).flatten()
        combined = np.concatenate([conf_back_flat, conf_seat_flat])
        all_confmat_data.append(combined)
    
    all_confmat_data = np.array(all_confmat_data)
    print(f"📐 ConfMat data shape: {all_confmat_data.shape}")
    
    # Fit StandardScaler on ConfMat data
    confmat_scaler_model = StandardScaler()
    normalized_confmat_data = confmat_scaler_model.fit_transform(all_confmat_data)
    
    print(f"✅ ConfMat normalization complete!")
    print(f"📊 Mean: {normalized_confmat_data.mean():.6f}, Std: {normalized_confmat_data.std():.6f}")
    
    # Reshape normalized ConfMat data
    normalized_confmat_frames = []
    for i, norm_vector in enumerate(normalized_confmat_data):
        conf_back = norm_vector[:conf_back_size].reshape(conf_back_shape)
        conf_seat = norm_vector[conf_back_size:].reshape(conf_seat_shape)
        
        normalized_confmat_frames.append({
            "posture": all_frames[i]["posture"],
            "conf_back": conf_back.tolist(),
            "conf_seat": conf_seat.tolist(),
            "user_id": user_id
        })
    
    # Count ConfMat frames per posture
    confmat_posture_counts = {}
    for frame in normalized_confmat_frames:
        posture = frame["posture"]
        confmat_posture_counts[posture] = confmat_posture_counts.get(posture, 0) + 1
    
    print(f"\n📊 Normalized ConfMat frames per posture:")
    for posture in sorted(confmat_posture_counts.keys()):
        print(f"   {posture}: {confmat_posture_counts[posture]} frames")
    
    # ========================================
    # NORMALIZE GTRACE DATA
    # ========================================
    print(f"\n{'='*60}")
    print(f"🔹 NORMALIZING GTRACE DATA")
    print(f"{'='*60}")
    
    all_gtrace_data = []
    for frame in all_frames:
        gtrace_back_flat = np.array(frame["gtrace_back"]).flatten()
        gtrace_seat_flat = np.array(frame["gtrace_seat"]).flatten()
        combined = np.concatenate([gtrace_back_flat, gtrace_seat_flat])
        all_gtrace_data.append(combined)
    
    all_gtrace_data = np.array(all_gtrace_data)
    print(f"📐 GTrace data shape: {all_gtrace_data.shape}")
    
    # Fit StandardScaler on GTrace data
    gtrace_scaler_model = StandardScaler()
    normalized_gtrace_data = gtrace_scaler_model.fit_transform(all_gtrace_data)
    
    print(f"✅ GTrace normalization complete!")
    print(f"📊 Mean: {normalized_gtrace_data.mean():.6f}, Std: {normalized_gtrace_data.std():.6f}")
    
    # Reshape normalized GTrace data
    normalized_gtrace_frames = []
    for i, norm_vector in enumerate(normalized_gtrace_data):
        gtrace_back = norm_vector[:gtrace_back_size].reshape(gtrace_back_shape)
        gtrace_seat = norm_vector[gtrace_back_size:].reshape(gtrace_seat_shape)
        
        normalized_gtrace_frames.append({
            "posture": all_frames[i]["posture"],
            "gtrace_back": gtrace_back.tolist(),
            "gtrace_seat": gtrace_seat.tolist(),
            "user_id": user_id
        })
    
    # Count GTrace frames per posture
    gtrace_posture_counts = {}
    for frame in normalized_gtrace_frames:
        posture = frame["posture"]
        gtrace_posture_counts[posture] = gtrace_posture_counts.get(posture, 0) + 1
    
    print(f"\n📊 Normalized GTrace frames per posture:")
    for posture in sorted(gtrace_posture_counts.keys()):
        print(f"   {posture}: {gtrace_posture_counts[posture]} frames")
    
    # ========================================
    # SAVE CONFMAT DATA
    # ========================================
    os.makedirs(normalized_confmat.path, exist_ok=True)
    
    with open(f"{normalized_confmat.path}/normalized_confmat_frames.json", "w") as f:
        json.dump(normalized_confmat_frames, f)
    
    confmat_metadata = {
        **metadata,
        "modality": "confmat",
        "channels": ["conf_back", "conf_seat"],
        "normalized_frame_count": len(normalized_confmat_frames),
        "posture_counts": confmat_posture_counts,
        "scaler_type": "StandardScaler",
        "feature_vector_size": all_confmat_data.shape[1],
        "channel_sizes": {"conf_back": conf_back_size, "conf_seat": conf_seat_size},
        "channel_shapes": {"conf_back": list(conf_back_shape), "conf_seat": list(conf_seat_shape)},
        "normalization_stats": {
            "mean": float(normalized_confmat_data.mean()),
            "std": float(normalized_confmat_data.std()),
            "min": float(normalized_confmat_data.min()),
            "max": float(normalized_confmat_data.max())
        }
    }
    
    with open(f"{normalized_confmat.path}/confmat_metadata.json", "w") as f:
        json.dump(confmat_metadata, f, indent=2)
    
    # Save ConfMat scaler
    os.makedirs(confmat_scaler.path, exist_ok=True)
    with open(f"{confmat_scaler.path}/confmat_scaler.pkl", "wb") as f:
        pickle.dump(confmat_scaler_model, f)
    
    confmat_scaler_metadata = {
        "user_id": user_id,
        "modality": "confmat",
        "scaler_type": "StandardScaler",
        "n_features": all_confmat_data.shape[1],
        "n_samples_fit": all_confmat_data.shape[0],
        "mean": confmat_scaler_model.mean_.tolist(),
        "scale": confmat_scaler_model.scale_.tolist(),
        "var": confmat_scaler_model.var_.tolist()
    }
    
    with open(f"{confmat_scaler.path}/confmat_scaler_metadata.json", "w") as f:
        json.dump(confmat_scaler_metadata, f, indent=2)
    
    # ========================================
    # SAVE GTRACE DATA
    # ========================================
    os.makedirs(normalized_gtrace.path, exist_ok=True)
    
    with open(f"{normalized_gtrace.path}/normalized_gtrace_frames.json", "w") as f:
        json.dump(normalized_gtrace_frames, f)
    
    gtrace_metadata = {
        **metadata,
        "modality": "gtrace",
        "channels": ["gtrace_back", "gtrace_seat"],
        "normalized_frame_count": len(normalized_gtrace_frames),
        "posture_counts": gtrace_posture_counts,
        "scaler_type": "StandardScaler",
        "feature_vector_size": all_gtrace_data.shape[1],
        "channel_sizes": {"gtrace_back": gtrace_back_size, "gtrace_seat": gtrace_seat_size},
        "channel_shapes": {"gtrace_back": list(gtrace_back_shape), "gtrace_seat": list(gtrace_seat_shape)},
        "normalization_stats": {
            "mean": float(normalized_gtrace_data.mean()),
            "std": float(normalized_gtrace_data.std()),
            "min": float(normalized_gtrace_data.min()),
            "max": float(normalized_gtrace_data.max())
        }
    }
    
    with open(f"{normalized_gtrace.path}/gtrace_metadata.json", "w") as f:
        json.dump(gtrace_metadata, f, indent=2)
    
    # Save GTrace scaler
    os.makedirs(gtrace_scaler.path, exist_ok=True)
    with open(f"{gtrace_scaler.path}/gtrace_scaler.pkl", "wb") as f:
        pickle.dump(gtrace_scaler_model, f)
    
    gtrace_scaler_metadata = {
        "user_id": user_id,
        "modality": "gtrace",
        "scaler_type": "StandardScaler",
        "n_features": all_gtrace_data.shape[1],
        "n_samples_fit": all_gtrace_data.shape[0],
        "mean": gtrace_scaler_model.mean_.tolist(),
        "scale": gtrace_scaler_model.scale_.tolist(),
        "var": gtrace_scaler_model.var_.tolist()
    }
    
    with open(f"{gtrace_scaler.path}/gtrace_scaler_metadata.json", "w") as f:
        json.dump(gtrace_scaler_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ NORMALIZATION COMPLETE FOR USER {user_id}")
    print(f"{'='*60}")
    print(f"💾 ConfMat: {len(normalized_confmat_frames)} frames saved")
    print(f"💾 GTrace: {len(normalized_gtrace_frames)} frames saved")
    print(f"💾 ConfMat scaler: {confmat_scaler.path}")
    print(f"💾 GTrace scaler: {gtrace_scaler.path}")