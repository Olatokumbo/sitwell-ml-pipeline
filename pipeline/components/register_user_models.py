from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Model, Output, Dataset

@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform"]
)
def register_user_models(
    user_id: str,
    gtrace_model: Input[Model],
    confmat_model: Input[Model],
    output_dataset: Output[Dataset],
):
    """Registers both GTrace and ConfMat models for a specific user."""
    from google.cloud import aiplatform
    import os
    import json

    project = os.getenv("GCLOUD_PROJECT_ID")
    region = os.getenv("GCLOUD_REGION")

    aiplatform.init(project=project, location=region)

    # Register the GTrace model
    gtrace_model = aiplatform.Model.upload(
        display_name=f"user_{user_id}_gtrace_model",
        artifact_uri=gtrace_model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        description=f"GTrace CNN model for user {user_id}"
    )

    # Register the ConfMat model
    confmat_model = aiplatform.Model.upload(
        display_name=f"user_{user_id}_confmat_model",
        artifact_uri=confmat_model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        description=f"ConfMat CNN model for user {user_id}"
    )

    # Deploy both models to endpoints
    gtrace_endpoint = gtrace_model.deploy(
        deployed_model_display_name=f"user_{user_id}_gtrace_endpoint",
        machine_type="e2-standard-2",
    )

    confmat_endpoint = confmat_model.deploy(
        deployed_model_display_name=f"user_{user_id}_confmat_endpoint",
        machine_type="e2-standard-2",
    )

    # Store everything for downstream use
    model_info = {
        "user_id": user_id,
        "gtrace_model_id": gtrace_model.resource_name,
        "confmat_model_id": confmat_model.resource_name,
        "gtrace_endpoint_id": gtrace_endpoint.resource_name,
        "confmat_endpoint_id": confmat_endpoint.resource_name,
    }

    with open(output_dataset.path, "w") as f:
        json.dump(model_info, f)