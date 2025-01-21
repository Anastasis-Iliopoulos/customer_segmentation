from sklearn.cluster import KMeans
import yaml
import logging
import transformations as TR
import utils as UT
import evaluation as EV
import os
import argparse
import json
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd


# Predefined mapping of model names to their corresponding classes
MODEL_MAPPING = {
    "kmeans": KMeans
}

logger = logging.getLogger(__name__)


def get_model(model_name, **kwargs):
    """
    Load model.

    Parameters
    ----------
    model_name : str
        The name of the model as defined in MODEL_MAPPING.
        
    Returns
    -------
    object
        Instantiated model.
    
    Raises
    ------
    ValueError
        If the model_name is not found in the `MODEL_MAPPING` dictionary.
    """
    try:
        
        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Model {model_name} is not supported.")
        model = MODEL_MAPPING[model_name]

        return model(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_name}: {e}")

def train_model(data, model_config):
    """
    Train and evaluate a machine learning pipeline for the specified model.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    model_config : dict
        Configuration for the model and its preprocessing pipeline.

    Returns
    -------
    tuple
        - Trained model object.
        - List of transformations applied to the data (for reproducibility).
        - Transformed data after applying transformations.
        - predicted data after fitting the model.
    
    Raises
    ------
    ValueError
        If an unknown transformation type is specified in the pipeline configuration.
    NotImplementedError
        If the model does not support either `transform` or `predict` methods for post-fitting transformation.
    """
    try:
        transformations = []
        random_state = model_config["params"]["random_state"]
        for transformation, params in model_config["pipeline"].items():
            if transformation == "standard_scaling":
                logger.info(f"Applying Standard Scaling to columns: {params}")
                scaler_pipeline = TR.apply_standard_scaling(params)
                data = scaler_pipeline.fit_transform(data)
                data = pd.DataFrame(
                    data, 
                    columns=scaler_pipeline.get_feature_names_out(), 
                )
                transformations.append(("standard_scaling", scaler_pipeline))
            elif transformation == "one_hot_encoding":
                logger.info(f"Applying One-Hot Encoding to columns: {params}")
                one_hot_pipeline = TR.apply_one_hot_encoding(params)
                data = one_hot_pipeline.fit_transform(data)
                data = pd.DataFrame(
                    data, 
                    columns=one_hot_pipeline.get_feature_names_out(), 
                )
                transformations.append(("one_hot_encoding", one_hot_pipeline))
            elif transformation == "pca_features":
                n_components = model_config["params"]["pca_n_components"]
                logger.info(f"Applying PCA to columns: {params} with pca_n_components={n_components}")
                pca_pipeline = TR.apply_pca(params, n_components, random_state)
                data = pca_pipeline.fit_transform(data)
                data = pd.DataFrame(
                    data, 
                    columns=pca_pipeline.get_feature_names_out(),
                )
                transformations.append(("pca", pca_pipeline))
            else:
                raise ValueError(f"Unknown transformation type: {transformation}")

        transformed_data = data
        model_name = model_config["model_mapping_key"]
        model_params = model_config["params"]["model_params"]
        logger.info(f"Initializing model: {model_name} with parameters: {model_params}")
        model = get_model(model_name, random_state=random_state, **model_params)
        model.fit(transformed_data)

        
        if hasattr(model, "predict"):
            logger.info("Predicting transformed_data using the model.")
            predicted_data = model.predict(transformed_data)
        else:
            raise NotImplementedError(f"The model {model_name} does not support predict method.")

        return model, transformations, transformed_data, predicted_data 
    except Exception as e:
        logger.error(f"Unexpected error in train_and_evaluate: {e}")
        raise

def parse_args():
    """
    Parse command-line arguments for train.py.
    """
    parser = argparse.ArgumentParser(description="Train a machine learning pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--override", type=str, help="JSON string containing key-value pairs eg. '{\"model.kmeans.params.n_clusters\": 8, \"model.name\": \"MyModel\", \"model.features\": [\"feat1\", \"feat2\"], \"model.enable_feature\": true}'")
    return parser.parse_args()

def run_training_pipeline(args_dict):
    """
    Run the training and evaluation pipeline.

    Parameters
    ----------
    args_dict : dict
        Parsed command-line arguments containing `config` and `override` values.
    """
    try:

        with open(args_dict["config"], "r") as file:
            config = yaml.safe_load(file)

        overrides = {}
        if args_dict["override"]:
            overrides = json.loads(args_dict["override"])
        config = UT.override_config(config, overrides)

        UT.setup_logger(config["general"]["logs_path"])
        data = UT.load_from_parquet(config["features"]["features_path"])
        mlflow.set_experiment(config["mlflow"]["mlflow_experiment_name"])
        model, transformations, transformed_data, predicted_data = train_model(data, config["model"]["kmeans"])

        input_example = transformed_data.iloc[:5]
        signature = infer_signature(input_example, model.predict(input_example))

        with mlflow.start_run():

            if not os.path.exists(os.path.dirname("./tmp/config.yaml")):
                os.makedirs(os.path.dirname("./tmp/config.yaml"), exist_ok=True)
            with open("./tmp/config.yaml", "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            mlflow.log_artifact("./tmp/config.yaml", artifact_path="configuration")
            os.remove("./tmp/config.yaml")

            for name, pipeline in transformations:
                pipeline_path = f"./tmp/{name}_pipeline.pkl"
                if os.path.dirname(pipeline_path):
                    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
                joblib.dump(pipeline, pipeline_path)
                logger.info(f"log artifact: {name}")
                mlflow.log_artifact(pipeline_path, artifact_path="preprocessing")
                os.remove(pipeline_path)

            logger.info(f"log model: {config['model']['kmeans']['model_mapping_key']}")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            logger.info(f"Starting Evaluation.")
            metrics = EV.evaluate_kmeans(model, transformed_data, config["model"]["kmeans"])
            logger.info(f"Evaluation metrics: {metrics}")
            for metric_name, metric_value in metrics.items():
                logger.info(f"log metric: {metric_name}")
                mlflow.log_metric(metric_name, metric_value)

            flattened_params = UT.flatten_dict(config["model"]["kmeans"]["params"])
            logger.info(f"log params: {flattened_params}")
            mlflow.log_params(flattened_params)

        logger.info("Training and evaluation completed. Artifacts and metrics logged to MLflow.")

    except Exception as e:
        logger.error(f"An error occurred during the training pipeline: {e}")
        raise

if __name__ == "__main__":

    args = parse_args()
    args_dict = vars(args)
    run_training_pipeline(args_dict)