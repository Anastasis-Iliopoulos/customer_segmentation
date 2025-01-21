from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

def evaluate_kmeans(model, transformed_data, model_config):
    """
    Evaluate a trained model using appropriate metrics.

    Parameters
    ----------
    model : object
        The trained model.
    transformed_data : np.array
        The processed data (after transformations).
    model_config : dict
        Configuration for the model and its evaluation.

    Returns
    -------
    dict
        A dictionary of evaluation metrics.
    """
    try:
        metrics = {}
        model_name = model_config["model_mapping_key"]

        if model_name == "kmeans":
            labels = model.labels_
            logger.info(f"Compute silhouette score.")
            silhouette_avg = silhouette_score(transformed_data, labels, n_jobs=-1)
            metrics["silhouette_score"] = silhouette_avg
            logger.info(f"Silhouette Score: {silhouette_avg:.3f}")\
            
            logger.info(f"Get inertia.")
            inertia = model.inertia_
            metrics["inertia"] = inertia
            logger.info(f"Inertia (Sum of Squared Distances): {inertia:.3f}")
        else:
            raise NotImplementedError(f"Evaluation not implemented for model: {model_name}")

        return metrics
    except Exception as e:
        logger.error(f"Unexpected error in evaluate_model: {e}")
        raise
