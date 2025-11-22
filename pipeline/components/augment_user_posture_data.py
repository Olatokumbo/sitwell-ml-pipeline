from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scipy", "scikit-image"]
)
def augment_user_posture_data(
    preprocessed_frames: Input[Dataset],
    augmented_frames: Output[Dataset],
    sample_size_per_posture: int = 10
):
    """Augment user-specific posture data with 4-channel structure"""
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
        frame["conf_back"] = np.array(frame["conf_back"])
        frame["conf_seat"] = np.array(frame["conf_seat"])
        frame["gtrace_back"] = np.array(frame["gtrace_back"])
        frame["gtrace_seat"] = np.array(frame["gtrace_seat"])
    
    # Augmentation functions for 2D data
    def add_noise(data, noise_ratio=0.5):
        data_std = np.std(data)
        noise_level = noise_ratio * data_std
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    def shift_data(data, shift_max=10):
        shift_x = np.random.randint(-shift_max, shift_max + 1)
        shift_y = np.random.randint(-shift_max, shift_max + 1)
        shifted = np.roll(data, shift_x, axis=0)
        shifted = np.roll(shifted, shift_y, axis=1)
        return np.clip(shifted, 0, 255)

    def rotate_data(data, angle):
        return np.clip(rotate(data, angle, reshape=False), 0, 255)

    def random_erasing(data, erase_prob=0.5, erase_size=0.1):
        if random.random() < erase_prob:
            data_copy = data.copy()
            h, w = data.shape
            eh = int(h * erase_size)
            ew = int(w * erase_size)
            top = np.random.randint(0, h - eh)
            left = np.random.randint(0, w - ew)
            data_copy[top:top + eh, left:left + ew] = 0
            return np.clip(data_copy, 0, 255)
        return np.clip(data, 0, 255)

    def elastic_transform(data, alpha, sigma):
        try:
            random_state = np.random.RandomState(None)
            shape = data.shape
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            transformed = map_coordinates(data, indices, order=1, mode='reflect').reshape(shape)
            return np.clip(transformed, 0, 255)
        except:
            return np.clip(data, 0, 255)

    def complex_augment_data(data, sample_size=10):
        augmented = []
        for _ in range(sample_size):
            d = data.copy()
            if random.choice([True, False]):
                d = shift_data(d, shift_max=np.random.randint(1, 6))
            if random.choice([True, False]):
                d = rotate_data(d, angle=np.random.uniform(-30, 30))
            if random.choice([True, False]):
                d = random_erasing(d)
            if random.choice([True, False]):
                d = elastic_transform(d, alpha=24, sigma=4)
            if random.choice([True, False]):
                d = add_noise(d)
            augmented.append(d)
        return augmented

    def combine_backrest_seat(back, seat):
        """Combine backrest and seat into single array"""
        return np.vstack((back, seat))

    # Generate augmented dataset
    augmented_dataset = []
    
    for i, frame in enumerate(all_frames):
        if i % 10 == 0:
            print(f"   Processing frame {i+1}/{len(all_frames)}")
            
        posture_name = frame["posture"]
        
        # Combine confMat and gtrace pairs (backrest + seat)
        conf_combined = combine_backrest_seat(frame["conf_back"], frame["conf_seat"])
        gtrace_combined = combine_backrest_seat(frame["gtrace_back"], frame["gtrace_seat"])
        
        # Augment each modality independently
        conf_augmented = complex_augment_data(conf_combined, sample_size=sample_size_per_posture)
        gtrace_augmented = complex_augment_data(gtrace_combined, sample_size=sample_size_per_posture)
        
        # Re-split and store
        for conf, gtrace in zip(conf_augmented, gtrace_augmented):
            # Split conf back into backrest and seat
            mid = conf.shape[0] // 2
            conf_back = conf[:mid, :]
            conf_seat = conf[mid:, :]
            
            # Split gtrace back into backrest and seat
            mid_g = gtrace.shape[0] // 2
            gtrace_back = gtrace[:mid_g, :]
            gtrace_seat = gtrace[mid_g:, :]
            
            augmented_dataset.append({
                "posture": posture_name,
                "conf_back": conf_back.tolist(),
                "conf_seat": conf_seat.tolist(),
                "gtrace_back": gtrace_back.tolist(),
                "gtrace_seat": gtrace_seat.tolist(),
                "user_id": user_id
            })
    
    print(f"✅ Augmentation complete for user {user_id}!")
    print(f"📊 Original dataset size: {len(all_frames)}")
    print(f"📊 Augmented dataset size: {len(augmented_dataset)}")
    
    # Count augmented frames per posture
    posture_counts = {}
    for frame in augmented_dataset:
        posture = frame["posture"]
        posture_counts[posture] = posture_counts.get(posture, 0) + 1
    
    print("\n📊 Augmented frames per posture:")
    for posture in sorted(posture_counts.keys()):
        print(f"   {posture}: {posture_counts[posture]} frames")
    
    # Convert numpy arrays back to lists for JSON storage
    for frame in all_frames:
        frame["conf_back"] = frame["conf_back"].tolist()
        frame["conf_seat"] = frame["conf_seat"].tolist()
        frame["gtrace_back"] = frame["gtrace_back"].tolist()
        frame["gtrace_seat"] = frame["gtrace_seat"].tolist()
    
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
        "augmented_posture_counts": posture_counts,
        "total_frames_after_augmentation": len(all_frames) + len(augmented_dataset),
        "channels": ["conf_back", "conf_seat", "gtrace_back", "gtrace_seat"]
    }
    
    with open(f"{augmented_frames.path}/augmentation_metadata.json", "w") as f:
        json.dump(augmentation_metadata, f, indent=2)
    
    print(f"💾 Saved augmented data for user {user_id}")
    print(f"📁 Output location: {augmented_frames.path}")