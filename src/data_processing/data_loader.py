import pandas as pd
import logging
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Loads a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {filepath}: {e}")
        raise

