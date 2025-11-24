from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Model

@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-firestore", "firebase-admin"]
)
def upload_scaler_to_firestore(
    gtrace_scaler: Input[Model],
    confmat_scaler: Input[Model],
    user_id: str
):
    """Upload scaler metadata to Firestore for inference"""
    import json
    import os
    from google.cloud import firestore
    
    print(f"📤 Uploading scaler metadata to Firestore for user: {user_id}")
    
    project = 'project-sitwell'
    
    # Initialize Firestore
    db = firestore.Client(project=project)
    
    # Load GTrace scaler metadata
    with open(f"{gtrace_scaler.path}/gtrace_scaler_metadata.json", "r") as f:
        gtrace_metadata = json.load(f)
    
    # Load ConfMat scaler metadata
    with open(f"{confmat_scaler.path}/confmat_scaler_metadata.json", "r") as f:
        confmat_metadata = json.load(f)
    
    print(f"✅ Loaded GTrace scaler: {gtrace_metadata['n_features']} features, {gtrace_metadata['n_samples_fit']} samples")
    print(f"✅ Loaded ConfMat scaler: {confmat_metadata['n_features']} features, {confmat_metadata['n_samples_fit']} samples")
    
    # Update calibration_sessions document with scaler data
    doc_ref = db.collection('calibration_sessions').document(user_id)
    
    update_data = {
        'gtrace_scaler': {
            'mean': gtrace_metadata['mean'],
            'scale': gtrace_metadata['scale'],
            'n_features': gtrace_metadata['n_features'],
            'n_samples_fit': gtrace_metadata['n_samples_fit']
        },
        'confmat_scaler': {
            'mean': confmat_metadata['mean'],
            'scale': confmat_metadata['scale'],
            'n_features': confmat_metadata['n_features'],
            'n_samples_fit': confmat_metadata['n_samples_fit']
        },
        'scaler_updated_at': firestore.SERVER_TIMESTAMP
    }
    
    doc_ref.update(update_data)
    
    print("✅ Uploaded scaler metadata to Firestore:")
    print(f"   Path: calibration_sessions/{user_id}")
    print(f"   GTrace features: {len(gtrace_metadata['mean'])}")
    print(f"   ConfMat features: {len(confmat_metadata['mean'])}")