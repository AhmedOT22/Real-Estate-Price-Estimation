import pytest
import pandas as pd
from src.data_processing import data_loader, features_preparation

def test_load_data():
    df = data_loader.load_data("data/raw/prices.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_split_data():
    df = data_loader.load_data("data/raw/prices.csv")
    X_train, X_test, y_train, y_test = features_preparation.split_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert X_train.shape[1] == X_test.shape[1]  # Same number of features
