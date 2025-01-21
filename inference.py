import argparse
import pandas as pd
import logging
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
import yaml
import joblib
import os
import transformations as TR
import utils as UT

logger = logging.getLogger(__name__)

def load_config(run_id, **kwargs):
    """
    Load the configuration file for a specific MLFlow run.

    Parameters
    ----------
    run_id : str
        The ID of the MLFlow run from which to retrieve the configuration file.
    **kwargs : dict, optional
        Additional keyword arguments passed to `mlflow.artifacts.download_artifacts`, 
        such as `dst_path` for specifying a custom download location.

    Returns
    -------
    dict
        A dictionary containing the configuration loaded from the YAML file.

    Notes
    -----
    - The configuration file is downloaded using `mlflow.artifacts.download_artifacts`.
    - After loading the configuration, the downloaded file is removed.
    """
    config_path = download_artifacts(run_id=run_id, artifact_path="configuration/config.yaml", **kwargs)
    logger.info(f"Download file: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"File Loaded: {config_path}")
    # os.remove(config_path)
    # logger.info(f"File removed from filesystem: {config_path}")
    return config

def load_transformers(run_id):
    """
    Load the transformers for the given run ID.

    Parameters
    ----------
    run_id : str
        The MLFlow run ID.

    Returns
    -------
    dict
        A dictionary of preprocessing transformers.
    """
    transformers = {}
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, "preprocessing")
    for artifact in artifacts:
        local_path = download_artifacts(run_id=run_id, artifact_path=artifact.path)
        step_name = artifact.path.split("/")[-1].replace("_pipeline.pkl", "")
        transformers[step_name] = joblib.load(local_path)
    return transformers

def apply_transformations(data, config, transformers):
    """
    Apply transformations to the data based on the configuration.

    Parameters
    ----------
    data : pd.DataFrame
        The input data to be transformed.
    config : dict
        The configuration dictionary containing transformation steps.
    transformers : dict
        The loaded preprocessing pipelines.

    Returns
    -------
    pd.DataFrame
        Transformed data.
    """

    data_index = data.index
    for step in config["model"]["kmeans"]["pipeline"]:
        if step == "standard_scaling":
            transformer = transformers.get("standard_scaling")
            if transformer:
                data = transformer.transform(data)
                data = pd.DataFrame(data, columns=transformer.get_feature_names_out(), index=data_index)
        elif step == "one_hot_encoding":
            transformer = transformers.get("one_hot_encoding")
            if transformer:
                data = transformer.transform(data)
                data = pd.DataFrame(data, columns=transformer.get_feature_names_out(), index=data_index)
        elif step == "pca_features":
            transformer = transformers.get("pca")
            if transformer:
                data = transformer.transform(data)
                data = pd.DataFrame(data, columns=transformer.get_feature_names_out(), index=data_index)
        else:
            raise ValueError(f"Unknown transformation step: {step}")
    return data

def get_predictions(run_id, new_data):
    """
    Get predictions using the model, configuration, and transformers for the given run ID.

    Parameters
    ----------
    run_id : str
        The MLFlow run ID.
    new_data : pd.DataFrame
        The new data to be clustered.

    Returns
    -------
    pd.DataFrame
        New data with cluster assignments.
    """
    config = load_config(run_id)
    transformers = load_transformers(run_id)
    transformed_data = apply_transformations(new_data, config, transformers)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    cluster_labels = model.predict(transformed_data)
    transformed_data["cluster"] = cluster_labels

    return transformed_data

def run_inference_pipeline(run_id, input_path, output_path):
    """
    Main function for running the inference pipeline.

    Parameters
    ----------
    run_id : str
        MLFlow run ID to load artifacts and configuration.
    input_path : str
        Path to the input Parquet file.
    output_path : str
        Path to save the output Parquet file.
    
    Returns
    -------
    None
    """
    try:
        input_data = UT.load_from_parquet(input_path)
        logger.info(f"Loaded input data with shape: {input_data.shape}")
        logger.info(f"Calculating predictions...")
        clustered_data = get_predictions(run_id, input_data)
        clustered_data = clustered_data.add_prefix("clustered_")
        final_data = input_data.join(clustered_data)
        UT.save_to_parquet(final_data, output_path)
        logger.info(f"Output data with shape ({clustered_data.shape}) saved to {output_path}.")

    except Exception as e:
        logger.error(f"An error occurred in the inference pipeline: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for customer segmentation.")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID to load artifacts.")
    parser.add_argument("--input_data", type=str, required=True, help="Path to the input parquet file.")
    parser.add_argument("--output_data", type=str, required=True, help="Path to save the predictions parquet file.")
    args = parser.parse_args()

    try:
        run_inference_pipeline(args.run_id, args.input_data, args.output_data)
    except Exception as e:
        logger.error(f"An error occurred in the inference pipeline: {e}")
        raise
