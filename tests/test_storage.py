import pytest
from src.models import storage
from sklearn.linear_model import LinearRegression
import numpy as np

def test_save_and_load_model(tmp_path):
    model = LinearRegression()
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    model.fit(X, y)

    path = tmp_path / "model.pkl"
    storage.save_model(model, str(path))
    loaded_model = storage.load_model(str(path))

    assert hasattr(loaded_model, "predict")
    assert callable(loaded_model.predict)
