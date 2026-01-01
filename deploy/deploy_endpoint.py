"""
Optional SageMaker deployment script for the Prophet CPI model.

WARNING:
- This script creates a SageMaker endpoint and WILL incur cost while running.
- Run only when you explicitly want to deploy.
- Always delete the endpoint immediately after testing.

The project is complete without running this script.
"""

import time
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


def main():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    region = sess.boto_region_name

    # Upload model artifact
    s3_model_uri = sess.upload_data(
        path="prophet-model.tar.gz",
        key_prefix="prophet-cpi/model"
    )
    print("Uploaded model to:", s3_model_uri)

    # Use PyTorch inference image as generic Python serving container
    image_uri = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="2.1.0",
        py_version="py310",
        image_scope="inference",
        instance_type="ml.m5.large"
    )
    print("Using image:", image_uri)

    model = Model(
        image_uri=image_uri,
        model_data=s3_model_uri,
        role=role,
        entry_point="inference.py",
        source_dir="sagemaker_prophet_inference"
    )

    endpoint_name = f"prophet-cpi-{int(time.time())}"
    print("Deploying endpoint:", endpoint_name)

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name
    )

    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    # Test inference once
    payload = {"ds": ["2024-01-01", "2024-02-01", "2024-03-01"]}
    response = predictor.predict(payload)
    print("Sample prediction:", response[:1])

    # Cleanup (IMPORTANT)
    sm = boto3.client("sagemaker")

    ep = sm.describe_endpoint(EndpointName=endpoint_name)
    cfg_name = ep["EndpointConfigName"]
    cfg = sm.describe_endpoint_config(EndpointConfigName=cfg_name)
    model_name = cfg["ProductionVariants"][0]["ModelName"]

    predictor.delete_endpoint()
    sm.delete_endpoint_config(EndpointConfigName=cfg_name)
    sm.delete_model(ModelName=model_name)

    print("Deleted endpoint, config, and model")


if __name__ == "__main__":
    main()