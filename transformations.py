import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def apply_one_hot_encoding(cols):
    """
    Create a one-hot encoding pipeline for categorical columns.

    Parameters
    ----------
    cols : dict
        dict of column names to apply one-hot encoding. Every values of keys are list. If a list if empty then category = 'auto' else category is predefined

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline for one-hot encoding.
    """
    try:
        one_hot_encoders = []
        one_hot_encoders = []

        for col, categories in cols.items():
            # or: if not categories
            if categories==[]:
                categories = 'auto'

            one_hot_encoders.append((
                f'one_hot_{col}',
                OneHotEncoder(categories=[categories] if categories != 'auto' else 'auto'),
                [col]
            ))

        pipeline = Pipeline([
            (
                'on_hot_encoding',
                ColumnTransformer(
                    transformers=one_hot_encoders, 
                    remainder='passthrough',
                    verbose_feature_names_out=False
                )
            )
        ])

        logger.info("Pipeline with one-hot encoding built successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Unexpected error while building one-hot encoding pipeline: {e}")
        raise

def apply_pca(cols, n_components, random_state):
    """
    Create a PCA pipeline for dimensionality reduction on specific columns.

    Parameters
    ----------
    cols : list
        List of column names to apply PCA on.
    n_components : int
        Number of PCA components to retain.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        ColumnTransformer that applies PCA to the specified columns.
    """
    try:
        pca = PCA(n_components=n_components, random_state=random_state)
        pipeline = ColumnTransformer(
            transformers=[
                ('pca', pca, cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        logger.info(f"PCA pipeline built successfully for columns {cols} with {n_components} components")
        return pipeline
    except Exception as e:
        logger.error(f"Unexpected error while building PCA pipeline: {e}")
        raise

def apply_standard_scaling(cols):
    """
    Create a standard scaling pipeline for numerical features.

    Parameters
    ----------
    cols : list
        List of column names to be scaled.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Pipeline for scaling numerical features.
    """
    try:
        scaler = StandardScaler()

        pipeline = ColumnTransformer(
            transformers=[
                ('scaler', scaler, cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        logger.info("Standard scaling pipeline built successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Unexpected error while building standard scaling pipeline: {e}")
        raise