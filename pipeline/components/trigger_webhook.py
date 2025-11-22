from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Dataset

@component(
    base_image="python:3.11",
    packages_to_install=["requests"]
)
def trigger_webhook(user_id: str, webhook_url: str, model_info: Input[Dataset]):
    """
    Triggers a webhook with the user ID as JSON payload.
    """
    import json
    import requests
    
    with open(model_info.path) as f:
        data = json.load(f)

    gtrace_id =  data["gtrace_model_id"].split("/")[5]
    confmat_id = data["confmat_model_id"].split("/")[5]
    
    # Create payload safely
    payload = json.dumps({"userId": user_id, "state": "PIPELINE_STATE_SUCCEEDED", "gtrace": gtrace_id, "confmat": confmat_id})

    # Set headers
    headers = {"Content-Type": "application/json"}

    # Send POST request
    response = requests.post(webhook_url, data=payload, headers=headers)

    print(f"Webhook response status: {response.status_code}")
    print(f"Webhook response body: {response.text}")

    # Optionally raise exception if webhook failed
    response.raise_for_status()