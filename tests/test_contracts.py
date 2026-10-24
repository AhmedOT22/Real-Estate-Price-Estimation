import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.data_processing.data_loader import load_data
from src.data_processing.features_preparation import split_data
from src.models.prediction import predict_single


def test_split_is_repeatable_disjoint_and_ordered():
    data = load_data('data/raw/prices.csv')
    a = split_data(data)
    b = split_data(data.sample(frac=1, random_state=1).sort_index())
    for first, second in zip(a, b):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)
    assert a[0].index.equals(b[0].index)
    assert a[1].index.equals(b[1].index)
    assert set(a[0].index).isdisjoint(a[1].index)
    assert len(a[0]) + len(a[1]) == len(data)


@pytest.mark.parametrize('change', ['missing', 'infinite', 'extra', 'age', 'type'])
def test_training_contract_rejects_invalid_values(change):
    data = load_data('data/raw/prices.csv')
    if change == 'missing':
        data.loc[0, 'price'] = np.nan
    elif change == 'infinite':
        data['sqft'] = data['sqft'].astype(float)
        data.loc[0, 'sqft'] = np.inf
    elif change == 'extra':
        data['accidental_target_copy'] = data['price']
    elif change == 'age':
        data.loc[0, 'property_age'] += 1
    else:
        data.loc[0, ['property_type_Bunglow', 'property_type_Condo']] = 1
    with pytest.raises(ValueError):
        split_data(data)


def test_prediction_accepts_numpy_and_rejects_ambiguous_inputs():
    model = LinearRegression().fit(pd.DataFrame({'a': [1., 2., 3.], 'b': [3., 1., 2.]}), [4., 3., 5.])
    assert predict_single(model, np.array([1., 3.])) == pytest.approx(4.)
    for values in ([], [1], [[1, 3], [2, 1]], [np.nan, 3], [np.inf, 3], ['text', 3]):
        with pytest.raises(ValueError):
            predict_single(model, values)
