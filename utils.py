import logging
import pandas as pd

logger = logging.getLogger(__name__)

def setup_logger(logs_path="./logs.log"):
    """
    Set up the global logging configuration.

    Configures logging to output to both the console and a log file.

    Parameters
    ----------
    logs_path : str, optional
        Path to the log file where logs will be written, by default "./logs.log".

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[
            logging.FileHandler(logs_path),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logger set up: {logs_path}")

def override_config(config, overrides):
    """
    Override YAML configuration with provided key-value pairs.

    Parameters
    ----------
    config : dict
        Original YAML configuration.
    overrides : dict
        Dictionary with overrides (e.g., {"model.kmeans.params.n_clusters": 8}).

    Returns
    -------
    dict
        Updated configuration dictionary.
    """

    for key_path, value in overrides.items():
        keys = key_path.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    return config

def flatten_dict(d):
    """
    Flatten a nested dictionary.
    
    Parameters
    ----------
        d : dict: 
            The dictionary to flatten.

    Returns
    -------
    dict 
        A flattened dictionary.
    """
    items = []
    for key, value in d.items():
        new_key = key
        if isinstance(value, dict):
            items.extend(flatten_dict(value).items())
        else:
            items.append((new_key, value))
    return dict(items)


def save_to_parquet(data, output_path):
    """
    Save data to a Parquet file.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing predictions and input data.
    output_path : str
        Path to save the Parquet file.
    
    Returns
    -------
    None
    """
    try:
        data.to_parquet(output_path, compression='gzip', index=True)
        logger.info(f"Output data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output data to {output_path}: {e}")
        raise

def load_from_parquet(imput_path):
    """
    Load input data from a Parquet file.

    Parameters
    ----------
    input_path : str
        Path to the input Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded input data as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the dataset cannot be read as a Pandas DataFrame.
    Exception
        For any other unexpected errors.
    """
    try:
        logger.info(f"Loading data from {imput_path}")
        data = pd.read_parquet(imput_path)
        logger.info(f"Data loaded successfully with shape: {data.shape}")
        return data
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error reading dataset: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading dataset: {e}")
        raise
