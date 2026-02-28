from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings
)
from azure.identity import DefaultAzureCredential
import os
import mlflow

def deploy_model(model_name="nsfw-detector", version="latest"):
    """Deploy model to Azure ML with A/B testing support"""
    
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP", "guardian-ai-prod"),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE", "guardian-ml-workspace-prod")
    )
    
    endpoint_name = f"{model_name}-endpoint"
    
    print(f"🚀 Deploying {model_name} to Azure ML...")
    
    # Create or update endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description=f"{model_name} inference endpoint",
        auth_mode="key",
        tags={"model": model_name, "environment": "production"}
    )
    
    print(f"🔧 Creating endpoint: {endpoint_name}")
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Get latest model version
    if version == "latest":
        model_versions = ml_client.models.list(name=model_name)
        latest_version = max([int(m.version) for m in model_versions])
        version = str(latest_version)
    
    print(f"📊 Using model version: {version}")
    
    # Deploy champion model (production)
    champion_deployment = ManagedOnlineDeployment(
        name="champion",
        endpoint_name=endpoint_name,
        model=f"{model_name}:{version}",
        instance_type="Standard_DS3_v2",
        instance_count=3,
        environment_variables={
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
            "MODEL_VERSION": version
        },
        liveness_probe=ProbeSettings(
            failure_threshold=3,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=30
        ),
        readiness_probe=ProbeSettings(
            failure_threshold=3,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=30
        )
    )
    
    print("🏆 Deploying champion model...")
    ml_client.online_deployments.begin_create_or_update(champion_deployment).result()
    
    # Set traffic to champion
    endpoint.traffic = {"champion": 90}
    
    # Deploy challenger model for A/B testing (if enabled)
    if os.getenv("ENABLE_AB_TESTING", "false").lower() == "true":
        print("🧪 Deploying challenger model for A/B testing...")
        
        challenger_deployment = ManagedOnlineDeployment(
            name="challenger",
            endpoint_name=endpoint_name,
            model=f"{model_name}:{version}",
            instance_type="Standard_DS3_v2",
            instance_count=1,
            environment_variables={
                "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
                "MODEL_VERSION": version,
                "DEPLOYMENT_TYPE": "challenger"
            }
        )
        
        ml_client.online_deployments.begin_create_or_update(challenger_deployment).result()
        
        # Split traffic for A/B testing
        traffic_percent = int(os.getenv("CHALLENGER_TRAFFIC_PERCENT", "10"))
        endpoint.traffic = {
            "champion": 100 - traffic_percent,
            "challenger": traffic_percent
        }
    
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Get endpoint details
    endpoint_details = ml_client.online_endpoints.get(endpoint_name)
    scoring_uri = endpoint_details.scoring_uri
    
    print(f"\n✅ Deployment successful!")
    print(f"🔗 Scoring URI: {scoring_uri}")
    print(f"📊 Traffic split: {endpoint.traffic}")
    
    return scoring_uri

def deploy_all_models():
    """Deploy all trained models"""
    models = ["nsfw-detector", "violence-detector"]
    
    endpoints = {}
    for model_name in models:
        try:
            uri = deploy_model(model_name)
            endpoints[model_name] = uri
        except Exception as e:
            print(f"❌ Failed to deploy {model_name}: {e}")
    
    return endpoints

if __name__ == "__main__":
    print("🎯 Guardian AI - Model Deployment Pipeline")
    print("="*50)
    
    endpoints = deploy_all_models()
    
    print("\n🎉 Deployment Summary:")
    for model, uri in endpoints.items():
        print(f"  {model}: {uri}")
    
    print("\n📝 Update your .env file with these endpoints!")
