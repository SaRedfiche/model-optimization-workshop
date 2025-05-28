"""
This script contains the updated endpoint deployment code for notebook 2.
It modifies the deployment process to run in parallel rather than sequentially.
"""

# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Dictionary to store endpoint names and deployment status
endpoint_names = {}
endpoint_status = {}

# First, prepare all model configurations
print("Preparing model configurations...")
model_configs = {}

for model_key, info in model_info.items():
    print(f"Preparing configuration for {model_key} model...")
    
    # Create a unique endpoint name with sanitized model key
    endpoint_name = f"model-opt-workshop-{sanitize_name(model_key)}-{int(time.time())}"
    endpoint_names[model_key] = endpoint_name
    endpoint_status[model_key] = "Preparing"
    
    # Create environment variables for the Hugging Face model
    env = {
        'HF_MODEL_ID': info["hub_model_id"],
        'HF_TASK': info["task"]
    }
    
    # Create a Hugging Face model
    huggingface_model = HuggingFaceModel(
        model_data=None,  # No model data, will use HF_MODEL_ID instead
        role=SAGEMAKER_ROLE_ARN,
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        env=env
    )
    
    # Store the model configuration
    model_configs[model_key] = huggingface_model
    print(f"Configuration prepared for {model_key} model")

# Now deploy all models in parallel
print("\nDeploying all models in parallel...")
deployment_futures = {}

for model_key, model in model_configs.items():
    print(f"Starting deployment for {model_key} model...")
    endpoint_status[model_key] = "Deploying"
    
    # Deploy the model to an endpoint asynchronously
    deployment_futures[model_key] = model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE_TYPE,
        endpoint_name=endpoint_names[model_key],
        wait=False  # Don't wait for deployment to complete
    )
    
    print(f"Deployment started for {model_key} model to endpoint: {endpoint_names[model_key]}")

# Monitor deployment status
print("\nMonitoring deployment status...")

def check_endpoint_status():
    """Check the status of all endpoints being deployed."""
    sagemaker_client = boto3.client('sagemaker')
    statuses = {}
    
    for model_key, endpoint_name in endpoint_names.items():
        try:
            response = sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            statuses[model_key] = response['EndpointStatus']
        except Exception as e:
            statuses[model_key] = f"Error: {str(e)}"
    
    return statuses

# Wait for all deployments to complete
all_completed = False
while not all_completed:
    current_statuses = check_endpoint_status()
    
    # Clear previous output
    clear_output(wait=True)
    
    # Print current status
    print("Current deployment statuses:")
    for model_key, status in current_statuses.items():
        print(f"{model_key}: {status}")
    
    # Check if all deployments are completed
    all_completed = all(status in ["InService", "Failed"] for status in current_statuses.values())
    
    if not all_completed:
        print("\nWaiting for 30 seconds before checking again...")
        time.sleep(30)

print("\nAll deployments have completed.")

# Store endpoint names for later use
%store endpoint_names
