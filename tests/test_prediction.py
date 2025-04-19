import pytest
from src.models import prediction, storage

def test_predict_single_flat_and_nested():
    model = storage.load_model("models/DT_Model.pkl")
    features = list(model.feature_names_in_)
    
    # Flat input
    flat_input = [2020, 3000, 1000, 3, 2, 1500, 2000, 5000, 0, 0, 0, 20, 1, 0]
    assert len(flat_input) == len(features)
    flat_pred = prediction.predict_single(model, flat_input)
    assert isinstance(flat_pred, (int, float))
    
    # Nested input
    nested_input = [[2020, 3000, 1000, 3, 2, 1500, 2000, 5000, 0, 0, 0, 20, 1, 0]]
    assert len(nested_input[0]) == len(features)
    nested_pred = prediction.predict_single(model, nested_input)
    assert isinstance(nested_pred, (int, float))
