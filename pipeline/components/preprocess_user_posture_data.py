from kfp.v2 import dsl
from kfp.v2.dsl import component, Output, Dataset

@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-storage", "pandas", "numpy", "pyarrow"]
)
def preprocess_user_posture_data(
    bucket_name: str,
    user_id: str,
    preprocessed_frames: Output[Dataset]
):
    """Load and preprocess Parquet files for a specific user"""
    from google.cloud import storage
    import numpy as np
    import pandas as pd
    import json
    import os
    
    print(f"👤 Processing calibration data for user: {user_id}")
    print(f"🔍 Loading data from bucket: {bucket_name}")
    
    # Connect to Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Find Parquet files for this specific user
    user_prefix = f"users/{user_id}/posture_data"
    blobs = list(bucket.list_blobs(prefix=user_prefix))
    parquet_files = [blob.name for blob in blobs if blob.name.endswith('.parquet')]
    
    print(f"📁 Found {len(parquet_files)} Parquet files for {user_id}:")
    for file in parquet_files:
        print(f"   📄 {file}")
    
    if len(parquet_files) == 0:
        raise ValueError(f"No Parquet files found for user {user_id} in {user_prefix}")
    
    # Download and process each Parquet file
    os.makedirs("./temp_data", exist_ok=True)
    all_frames = []
    posture_counts = {}
    
    for blob_name in parquet_files:
        # Extract posture code from filename (e.g., "01-uuid.parquet" -> "uuid")
        filename = os.path.basename(blob_name)
        posture_code = filename.replace('.parquet', '').split("-", 1)[1]
        
        # Download Parquet file
        local_parquet_path = f"./temp_data/{filename}"
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_parquet_path)
        
        print(f"📥 Processing {posture_code}...")
        
        # Read Parquet file
        df = pd.read_parquet(local_parquet_path)
        
        print(f"   📊 Parquet shape: {df.shape}")
        print(f"   📋 Columns: {list(df.columns)}")
        print(f"   📋 Column types: {df.dtypes.to_dict()}")
        
        num_frames = len(df)
        print(f"   🔄 Extracting {num_frames} frames from {posture_code}...")
        
        # Extract frames for this posture
        for idx, row in df.iterrows():
            try:
                # Parse string columns as JSON (they're stored as VARCHAR in Parquet)
                # The data is stored as string representation like "[[0.0, 0.0, ...], ...]"
                def parse_sensor_data(data):
                    """Parse sensor data from string or array format"""
                    if isinstance(data, str):
                        # Parse JSON string
                        parsed = json.loads(data)
                        arr = np.array(parsed)
                    else:
                        # Already an array or list
                        arr = np.array(data)
                    
                    # Ensure it's 32x32
                    if arr.shape == (32, 32):
                        return arr
                    elif arr.size == 1024:
                        return arr.reshape(32, 32)
                    else:
                        raise ValueError(f"Unexpected shape: {arr.shape} (size={arr.size})")
                
                conf_back_2d = parse_sensor_data(row['conf_back'])
                conf_seat_2d = parse_sensor_data(row['conf_seat'])
                gtrace_back_2d = parse_sensor_data(row['gtrace_back'])
                gtrace_seat_2d = parse_sensor_data(row['gtrace_seat'])
                
                all_frames.append({
                    "posture": posture_code,
                    "conf_back": conf_back_2d.tolist(),
                    "conf_seat": conf_seat_2d.tolist(),
                    "gtrace_back": gtrace_back_2d.tolist(),
                    "gtrace_seat": gtrace_seat_2d.tolist(),
                    "user_id": user_id
                })
                
                # Log first frame for verification
                if idx == 0:
                    print(f"   ✓ First frame parsed successfully:")
                    print(f"      conf_back shape: {conf_back_2d.shape}")
                    print(f"      conf_back sample: {conf_back_2d[0, :5]}")
                
            except Exception as e:
                print(f"   ⚠️  Error processing frame {idx} in {posture_code}:")
                print(f"      Error: {e}")
                print(f"      conf_back type: {type(row['conf_back'])}")
                if isinstance(row['conf_back'], str):
                    print(f"      conf_back string preview: {row['conf_back'][:200]}...")
                raise
        
        posture_counts[posture_code] = num_frames
        print(f"   ✅ Processed {posture_code}: {num_frames} frames")
    
    print(f"\n🎉 Total frames processed for {user_id}: {len(all_frames)}")
    
    # Display summary
    print("\n📊 Frames per posture:")
    for posture in sorted(posture_counts.keys()):
        count = posture_counts[posture]
        print(f"   {posture}: {count} frames")
    
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
        "sensor_channels": 4,  # conf_back, conf_seat, gtrace_back, gtrace_seat
        "sensor_dimensions": f"{sensor_rows}x{sensor_cols}",
        "posture_timestamp": str(np.datetime64('now')),
        "frame_structure": {
            "conf_back": f"ConfMat_Back pressure sensors ({sensor_rows}x{sensor_cols})",
            "conf_seat": f"ConfMat_Seat pressure sensors ({sensor_rows}x{sensor_cols})", 
            "gtrace_back": f"GTrace_Back sensors ({sensor_rows}x{sensor_cols})",
            "gtrace_seat": f"GTrace_Seat sensors ({sensor_rows}x{sensor_cols})",
            "posture": "User-specific posture classification"
        }
    }
    
    with open(f"{preprocessed_frames.path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ User-specific preprocessing complete!")
    print(f"💾 Saved {len(all_frames)} processed frames for user {user_id}")
    print(f"📁 Output location: {preprocessed_frames.path}")