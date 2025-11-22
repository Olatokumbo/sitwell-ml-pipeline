from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "numpy", "scikit-learn"]
)
def split_train_val_test_data(
    normalized_confmat: Input[Dataset],
    normalized_gtrace: Input[Dataset],
    confmat_train: Output[Dataset],
    confmat_val: Output[Dataset],
    confmat_test: Output[Dataset],
    gtrace_train: Output[Dataset],
    gtrace_val: Output[Dataset],
    gtrace_test: Output[Dataset],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
):
    """
    Split both ConfMat and GTrace normalized data into stratified train/validation/test sets.
    
    This ensures:
    - Same random seed for reproducibility across both modalities
    - Stratified split (proportional representation of each posture class)
    - Separate datasets for training two independent CNN models
    
    Args:
        normalized_confmat: Normalized ConfMat data from normalization step
        normalized_gtrace: Normalized GTrace data from normalization step
        confmat_train/val/test: Output datasets for ConfMat CNN
        gtrace_train/val/test: Output datasets for GTrace CNN
        train_split: Proportion for training (default 0.7 = 70%)
        val_split: Proportion for validation (default 0.15 = 15%)
        test_split: Proportion for testing (default 0.15 = 15%)
        random_seed: Seed for reproducibility
    """
    import numpy as np
    import json
    import os
    
    # ========================================
    # VALIDATE SPLIT RATIOS
    # ========================================
    total_split = train_split + val_split + test_split
    if not np.isclose(total_split, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
    
    print(f"\n{'='*60}")
    print(f"TRAIN/VAL/TEST SPLIT")
    print(f"{'='*60}")
    print(f"🎯 Split ratios:")
    print(f"   Train: {train_split*100:.1f}%")
    print(f"   Validation: {val_split*100:.1f}%")
    print(f"   Test: {test_split*100:.1f}%")
    print(f"🌱 Random seed: {random_seed}")
    
    # ========================================
    # HELPER FUNCTION: STRATIFIED SPLIT
    # ========================================
    def stratified_split(frames, train_ratio, val_ratio, test_ratio, seed):
        """Perform stratified split by posture class"""
        # Group frames by posture
        posture_groups = {}
        for frame in frames:
            posture = frame["posture"]
            if posture not in posture_groups:
                posture_groups[posture] = []
            posture_groups[posture].append(frame)
        
        train_frames = []
        val_frames = []
        test_frames = []
        
        np.random.seed(seed)
        
        print(f"\n   Splitting by posture:")
        for posture in sorted(posture_groups.keys()):
            frames_in_posture = posture_groups[posture]
            n_frames = len(frames_in_posture)
            
            # Shuffle frames for this posture
            indices = np.random.permutation(n_frames)
            shuffled = [frames_in_posture[i] for i in indices]
            
            # Calculate split indices
            train_end = int(n_frames * train_ratio)
            val_end = train_end + int(n_frames * val_ratio)
            
            # Split
            train_posture = shuffled[:train_end]
            val_posture = shuffled[train_end:val_end]
            test_posture = shuffled[val_end:]
            
            train_frames.extend(train_posture)
            val_frames.extend(val_posture)
            test_frames.extend(test_posture)
            
            print(f"      {posture}: {len(train_posture)} train, {len(val_posture)} val, {len(test_posture)} test")
        
        # Shuffle combined datasets
        np.random.shuffle(train_frames)
        np.random.shuffle(val_frames)
        np.random.shuffle(test_frames)
        
        return train_frames, val_frames, test_frames
    
    # ========================================
    # HELPER FUNCTION: COUNT POSTURES
    # ========================================
    def count_postures(frames):
        """Count frames per posture"""
        counts = {}
        for frame in frames:
            posture = frame["posture"]
            counts[posture] = counts.get(posture, 0) + 1
        return counts
    
    # ========================================
    # SPLIT CONFMAT DATA
    # ========================================
    print(f"\n{'='*60}")
    print(f"🔹 CONFMAT DATA SPLIT")
    print(f"{'='*60}")
    
    # Load ConfMat data
    with open(f"{normalized_confmat.path}/normalized_confmat_frames.json", "r") as f:
        confmat_frames = json.load(f)
    
    with open(f"{normalized_confmat.path}/confmat_metadata.json", "r") as f:
        confmat_metadata = json.load(f)
    
    user_id = confmat_metadata["user_id"]
    print(f"👤 User: {user_id}")
    print(f"📊 Total ConfMat frames: {len(confmat_frames)}")
    
    # Perform stratified split
    confmat_train_frames, confmat_val_frames, confmat_test_frames = stratified_split(
        confmat_frames, train_split, val_split, test_split, random_seed
    )
    
    print(f"\n✅ ConfMat split complete:")
    print(f"   📊 Train: {len(confmat_train_frames)} frames ({len(confmat_train_frames)/len(confmat_frames)*100:.1f}%)")
    print(f"   📊 Val: {len(confmat_val_frames)} frames ({len(confmat_val_frames)/len(confmat_frames)*100:.1f}%)")
    print(f"   📊 Test: {len(confmat_test_frames)} frames ({len(confmat_test_frames)/len(confmat_frames)*100:.1f}%)")
    
    # Count postures in each split
    confmat_train_counts = count_postures(confmat_train_frames)
    confmat_val_counts = count_postures(confmat_val_frames)
    confmat_test_counts = count_postures(confmat_test_frames)
    
    # ========================================
    # SPLIT GTRACE DATA
    # ========================================
    print(f"\n{'='*60}")
    print(f"🔹 GTRACE DATA SPLIT")
    print(f"{'='*60}")
    
    # Load GTrace data
    with open(f"{normalized_gtrace.path}/normalized_gtrace_frames.json", "r") as f:
        gtrace_frames = json.load(f)
    
    with open(f"{normalized_gtrace.path}/gtrace_metadata.json", "r") as f:
        gtrace_metadata = json.load(f)
    
    print(f"👤 User: {user_id}")
    print(f"📊 Total GTrace frames: {len(gtrace_frames)}")
    
    # Perform stratified split
    gtrace_train_frames, gtrace_val_frames, gtrace_test_frames = stratified_split(
        gtrace_frames, train_split, val_split, test_split, random_seed
    )
    
    print(f"\n✅ GTrace split complete:")
    print(f"   📊 Train: {len(gtrace_train_frames)} frames ({len(gtrace_train_frames)/len(gtrace_frames)*100:.1f}%)")
    print(f"   📊 Val: {len(gtrace_val_frames)} frames ({len(gtrace_val_frames)/len(gtrace_frames)*100:.1f}%)")
    print(f"   📊 Test: {len(gtrace_test_frames)} frames ({len(gtrace_test_frames)/len(gtrace_frames)*100:.1f}%)")
    
    # Count postures in each split
    gtrace_train_counts = count_postures(gtrace_train_frames)
    gtrace_val_counts = count_postures(gtrace_val_frames)
    gtrace_test_counts = count_postures(gtrace_test_frames)
    
    # ========================================
    # SAVE CONFMAT TRAIN DATA
    # ========================================
    print(f"\n💾 Saving ConfMat splits...")
    
    os.makedirs(confmat_train.path, exist_ok=True)
    with open(f"{confmat_train.path}/confmat_train_frames.json", "w") as f:
        json.dump(confmat_train_frames, f)
    
    confmat_train_metadata = {
        **confmat_metadata,
        "split_type": "train",
        "n_frames": len(confmat_train_frames),
        "split_ratio": train_split,
        "posture_counts": confmat_train_counts,
        "random_seed": random_seed
    }
    with open(f"{confmat_train.path}/confmat_train_metadata.json", "w") as f:
        json.dump(confmat_train_metadata, f, indent=2)
    
    # ========================================
    # SAVE CONFMAT VAL DATA
    # ========================================
    os.makedirs(confmat_val.path, exist_ok=True)
    with open(f"{confmat_val.path}/confmat_val_frames.json", "w") as f:
        json.dump(confmat_val_frames, f)
    
    confmat_val_metadata = {
        **confmat_metadata,
        "split_type": "validation",
        "n_frames": len(confmat_val_frames),
        "split_ratio": val_split,
        "posture_counts": confmat_val_counts,
        "random_seed": random_seed
    }
    with open(f"{confmat_val.path}/confmat_val_metadata.json", "w") as f:
        json.dump(confmat_val_metadata, f, indent=2)
    
    # ========================================
    # SAVE CONFMAT TEST DATA
    # ========================================
    os.makedirs(confmat_test.path, exist_ok=True)
    with open(f"{confmat_test.path}/confmat_test_frames.json", "w") as f:
        json.dump(confmat_test_frames, f)
    
    confmat_test_metadata = {
        **confmat_metadata,
        "split_type": "test",
        "n_frames": len(confmat_test_frames),
        "split_ratio": test_split,
        "posture_counts": confmat_test_counts,
        "random_seed": random_seed
    }
    with open(f"{confmat_test.path}/confmat_test_metadata.json", "w") as f:
        json.dump(confmat_test_metadata, f, indent=2)
    
    # ========================================
    # SAVE GTRACE TRAIN DATA
    # ========================================
    print(f"💾 Saving GTrace splits...")
    
    os.makedirs(gtrace_train.path, exist_ok=True)
    with open(f"{gtrace_train.path}/gtrace_train_frames.json", "w") as f:
        json.dump(gtrace_train_frames, f)
    
    gtrace_train_metadata = {
        **gtrace_metadata,
        "split_type": "train",
        "n_frames": len(gtrace_train_frames),
        "split_ratio": train_split,
        "posture_counts": gtrace_train_counts,
        "random_seed": random_seed
    }
    with open(f"{gtrace_train.path}/gtrace_train_metadata.json", "w") as f:
        json.dump(gtrace_train_metadata, f, indent=2)
    
    # ========================================
    # SAVE GTRACE VAL DATA
    # ========================================
    os.makedirs(gtrace_val.path, exist_ok=True)
    with open(f"{gtrace_val.path}/gtrace_val_frames.json", "w") as f:
        json.dump(gtrace_val_frames, f)
    
    gtrace_val_metadata = {
        **gtrace_metadata,
        "split_type": "validation",
        "n_frames": len(gtrace_val_frames),
        "split_ratio": val_split,
        "posture_counts": gtrace_val_counts,
        "random_seed": random_seed
    }
    with open(f"{gtrace_val.path}/gtrace_val_metadata.json", "w") as f:
        json.dump(gtrace_val_metadata, f, indent=2)
    
    # ========================================
    # SAVE GTRACE TEST DATA
    # ========================================
    os.makedirs(gtrace_test.path, exist_ok=True)
    with open(f"{gtrace_test.path}/gtrace_test_frames.json", "w") as f:
        json.dump(gtrace_test_frames, f)
    
    gtrace_test_metadata = {
        **gtrace_metadata,
        "split_type": "test",
        "n_frames": len(gtrace_test_frames),
        "split_ratio": test_split,
        "posture_counts": gtrace_test_counts,
        "random_seed": random_seed
    }
    with open(f"{gtrace_test.path}/gtrace_test_metadata.json", "w") as f:
        json.dump(gtrace_test_metadata, f, indent=2)
    
    # ========================================
    # CREATE SUMMARY
    # ========================================
    summary = {
        "user_id": user_id,
        "random_seed": random_seed,
        "split_ratios": {
            "train": train_split,
            "validation": val_split,
            "test": test_split
        },
        "confmat": {
            "total_frames": len(confmat_frames),
            "train": {"n_frames": len(confmat_train_frames), "posture_counts": confmat_train_counts},
            "validation": {"n_frames": len(confmat_val_frames), "posture_counts": confmat_val_counts},
            "test": {"n_frames": len(confmat_test_frames), "posture_counts": confmat_test_counts}
        },
        "gtrace": {
            "total_frames": len(gtrace_frames),
            "train": {"n_frames": len(gtrace_train_frames), "posture_counts": gtrace_train_counts},
            "validation": {"n_frames": len(gtrace_val_frames), "posture_counts": gtrace_val_counts},
            "test": {"n_frames": len(gtrace_test_frames), "posture_counts": gtrace_test_counts}
        }
    }
    
    # Save summary to all output directories
    for path in [confmat_train.path, confmat_val.path, confmat_test.path,
                 gtrace_train.path, gtrace_val.path, gtrace_test.path]:
        with open(f"{path}/split_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ SPLIT COMPLETE FOR USER {user_id}")
    print(f"{'='*60}")
    print(f"✨ ConfMat datasets ready for CNN training")
    print(f"✨ GTrace datasets ready for CNN training")
    print(f"📁 All splits saved with metadata and summary")