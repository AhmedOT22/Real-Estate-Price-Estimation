import pandas as pd
import logging
from sklearn.model_selection import train_test_split

def split_data(df, target='price', stratify_col='property_type_Bunglow'):
    """
    Splits the dataset into train and test sets.

    Args:
        df (pd.DataFrame): Full dataset.
        target (str): Name of the target column.
        stratify_col (str): Column used for stratified splitting.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    try:
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X[stratify_col])
        logging.info(f"Data split into train and test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        raise