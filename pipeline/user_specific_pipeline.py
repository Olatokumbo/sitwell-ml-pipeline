# user_specific_pipeline.py - Personalized posture pipeline for individual users
from kfp.v2 import dsl, compiler
from kfp.v2.dsl import component, pipeline, Input, Output, Dataset, Model
from components import (preprocess_user_posture_data, augment_user_posture_data, normalize_confmat_and_gtrace_data, split_train_val_test_data, register_user_models, train_confmat_cnn, train_gtrace_cnn, trigger_webhook)

@pipeline(
    name="user-specific-posture-pipeline",
    description="Personalized posture classification pipeline for individual users"
)
def user_specific_posture_pipeline(
    bucket_name: str,
    user_id: str,
    augmentation_samples: int = 1,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    cnn_epochs: int = 30,
    cnn_batch_size: int = 8
):
    """
    Complete pipeline for user-specific posture classification with dual CNN models.
    
    Pipeline Flow:
    1. Preprocess NPZ files (SP01-SP19) for the user
    2. Augment data independently for ConfMat and GTrace
    3. Normalize both modalities separately
    4. Split into train/val/test sets
    5. Train ConfMat CNN model
    6. Train GTrace CNN model
    
    Args:
        bucket_name: GCS bucket containing user data
        user_id: User identifier (e.g., "janusz-kulon")
        augmentation_samples: Number of augmented samples per frame
        train_split: Training set proportion (default 0.7)
        val_split: Validation set proportion (default 0.15)
        test_split: Test set proportion (default 0.15)
        random_seed: Random seed for reproducibility
        cnn_epochs: Number of training epochs for CNNs
        cnn_batch_size: Batch size for CNN training
    """
    
    print(f"Starting pipeline for user: {user_id}")
    
    preprocess_task = preprocess_user_posture_data(
        bucket_name=bucket_name,
        user_id=user_id
    )
    preprocess_task.set_display_name("Preprocess NPZ Data")
    
    augment_task = augment_user_posture_data(
        preprocessed_frames=preprocess_task.outputs["preprocessed_frames"],
        sample_size_per_posture=augmentation_samples
    )
    augment_task.set_display_name("Augment Posture Data")
    augment_task.after(preprocess_task)
    
    
    normalize_task = normalize_confmat_and_gtrace_data(
        augmented_frames=augment_task.outputs["augmented_frames"]
    )
    normalize_task.set_display_name("Normalize ConfMat & GTrace")
    normalize_task.after(augment_task)
    normalize_task.set_memory_limit('32G')
    normalize_task.set_memory_request('8G')
    normalize_task.set_cpu_limit('4')
    normalize_task.set_cpu_request('2')
    
    
    split_task = split_train_val_test_data(
        normalized_confmat=normalize_task.outputs["normalized_confmat"],
        normalized_gtrace=normalize_task.outputs["normalized_gtrace"],
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )
    split_task.set_display_name("Split Train/Val/Test")
    split_task.after(normalize_task)
    
    
    train_confmat_task = train_confmat_cnn(
        confmat_train=split_task.outputs["confmat_train"],
        confmat_val=split_task.outputs["confmat_val"],
        confmat_test=split_task.outputs["confmat_test"],
        epochs=cnn_epochs,
        batch_size=cnn_batch_size
    )
    train_confmat_task.set_display_name("Train ConfMat CNN")
    train_confmat_task.after(split_task)
    
    # Set GPU for ConfMat training
    train_confmat_task.set_memory_limit('32G')
    train_confmat_task.set_cpu_limit('8')
    
    train_gtrace_task = train_gtrace_cnn(
        gtrace_train=split_task.outputs["gtrace_train"],
        gtrace_val=split_task.outputs["gtrace_val"],
        gtrace_test=split_task.outputs["gtrace_test"],
        epochs=cnn_epochs,
        batch_size=cnn_batch_size
    )
    train_gtrace_task.set_display_name("Train GTrace CNN")
    train_gtrace_task.after(split_task)
    
    # Set GPU for GTrace training
    train_gtrace_task.set_memory_limit('32G')
    train_gtrace_task.set_cpu_limit('8')

    
    register_task = register_user_models(
        user_id=user_id,
        gtrace_model=train_gtrace_task.outputs['gtrace_cnn_model'],
        confmat_model=train_confmat_task.outputs['confmat_cnn_model']
    )
    
    webhook_task = trigger_webhook(
        user_id=user_id,
        webhook_url="https://pipelinewebhook-hfarmdvsyq-uc.a.run.app",
        model_info=register_task.outputs['output_dataset']
    )
    

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
