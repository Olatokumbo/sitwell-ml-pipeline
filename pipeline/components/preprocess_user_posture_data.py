from kfp.v2 import dsl
from kfp.v2.dsl import component, Output, Dataset

@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-storage", "pandas", "numpy"]
)
def preprocess_user_posture_data(
    bucket_name: str,
    user_id: str,
    preprocessed_frames: Output[Dataset]
):
    """Load and preprocess NPZ files for a specific user"""
    from google.cloud import storage
    import numpy as np
    import json
    import os
    
    print(f"👤 Processing calibration data for user: {user_id}")
    print(f"🔍 Loading data from bucket: {bucket_name}")
    
    # Connect to Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Find NPZ files for this specific user
    user_prefix = f"users/{user_id}/posture_data"
    blobs = list(bucket.list_blobs(prefix=user_prefix))
    npz_files = [blob.name for blob in blobs if blob.name.endswith('.npz')]
    
    print(f"📁 Found {len(npz_files)} NPZ files for {user_id}:")
    for file in npz_files:
        print(f"   📄 {file}")
    
    if len(npz_files) == 0:
        raise ValueError(f"No NPZ files found for user {user_id} in {user_prefix}")
    
    # Expected posture codes
    expected_postures = [f"SP{i:02d}" for i in range(1, 20)]  # SP01 to SP19
    
    # Download and process each NPZ file
    os.makedirs("./temp_data", exist_ok=True)
    all_frames = []
    posture_counts = {}
    
    for blob_name in npz_files:
        # Extract posture code from filename (e.g., "SP01.npz" -> "SP01")
        filename = os.path.basename(blob_name)
        posture_code = filename.replace('.npz', '')
        
        # Skip if not a recognized posture
        if posture_code not in expected_postures:
            print(f"   ⚠️  Skipping unrecognized file: {filename}")
            continue
        
        # Download NPZ file
        local_npz_path = f"./temp_data/{filename}"
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_npz_path)
        
        print(f"📥 Processing {posture_code}...")
        
        # Load NPZ file
        npz_data = np.load(local_npz_path, allow_pickle=True)
        
        # Extract data array - adjust key based on your NPZ structure
        # Common possibilities: 'data', 'arr_0', or the actual array
        if 'data' in npz_data:
            data_array = npz_data['data']
        elif 'arr_0' in npz_data:
            data_array = npz_data['arr_0']
        else:
            # Take the first available array
            data_array = npz_data[list(npz_data.keys())[0]]
        
        # Verify shape: should be (time, sensor, rows, cols)
        if len(data_array.shape) != 4:
            print(f"   ⚠️  Warning: Unexpected shape {data_array.shape} for {posture_code}")
        
        num_frames = data_array.shape[0]
        print(f"   🔄 Extracting {num_frames} frames from {posture_code}...")
        
        # Extract frames for this posture
        for t in range(num_frames):
            all_frames.append({
                "posture": posture_code,
                "conf_back": data_array[t, 0, :, :].tolist(),   # ConfMat_Back
                "conf_seat": data_array[t, 1, :, :].tolist(),   # ConfMat_Seat
                "gtrace_back": data_array[t, 2, :, :].tolist(), # GTrace_Back
                "gtrace_seat": data_array[t, 3, :, :].tolist(), # GTrace_Seat
                "user_id": user_id
            })
        
        posture_counts[posture_code] = num_frames
        print(f"   ✅ Processed {posture_code}: {num_frames} frames")
        
        # Close NPZ file
        npz_data.close()
    
    print(f"\n🎉 Total frames processed for {user_id}: {len(all_frames)}")
    
    # Display summary
    print("\n📊 Frames per posture:")
    for posture in sorted(posture_counts.keys()):
        count = posture_counts[posture]
        print(f"   {posture}: {count} frames")
    
    # Check for missing postures
    found_postures = set(posture_counts.keys())
    missing_postures = set(expected_postures) - found_postures
    if missing_postures:
        print(f"\n⚠️  Missing postures: {sorted(missing_postures)}")
    
    # Save processed frames
    os.makedirs(preprocessed_frames.path, exist_ok=True)
    
    with open(f"{preprocessed_frames.path}/all_frames.json", "w") as f:
        json.dump(all_frames, f)
    
    # Get sensor dimensions from the first frame
    if all_frames:
        sensor_rows = len(all_frames[0]["conf_back"])
        sensor_cols = len(all_frames[0]["conf_back"][0])
    else:
        sensor_rows, sensor_cols = 32, 32  # Default assumption
    
    # Save metadata with user info
    metadata = {
        "user_id": user_id,
        "total_frames": len(all_frames),
        "posture_counts": posture_counts,
        "num_postures": len(posture_counts),
        "expected_postures": expected_postures,
        "found_postures": sorted(list(found_postures)),
        "missing_postures": sorted(list(missing_postures)),
        "sensor_channels": 4,  # conf_back, conf_seat, gtrace_back, gtrace_seat
        "sensor_dimensions": f"{sensor_rows}x{sensor_cols}",
        "posture_timestamp": str(np.datetime64('now')),
        "frame_structure": {
            "conf_back": f"ConfMat_Back pressure sensors ({sensor_rows}x{sensor_cols})",
            "conf_seat": f"ConfMat_Seat pressure sensors ({sensor_rows}x{sensor_cols})", 
            "gtrace_back": f"GTrace_Back sensors ({sensor_rows}x{sensor_cols})",
            "gtrace_seat": f"GTrace_Seat sensors ({sensor_rows}x{sensor_cols})",
            "posture": "User-specific posture classification (SP01-SP19)"
        }
    }
    
    with open(f"{preprocessed_frames.path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ User-specific preprocessing complete!")
    print(f"💾 Saved {len(all_frames)} processed frames for user {user_id}")
    print(f"📁 Output location: {preprocessed_frames.path}")