"""Single-row prediction with an explicit feature-order contract."""
import numpy as np
import pandas as pd


def predict_single(model, input_values):
    try:
        values = np.asarray(input_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Prediction inputs must be numeric.') from error
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape != (1, len(model.feature_names_in_)):
        raise ValueError(f'Expected one property with {len(model.feature_names_in_)} feature values.')
    if not np.isfinite(values).all():
        raise ValueError('Prediction inputs must be finite, with no missing values.')
    frame = pd.DataFrame(values, columns=model.feature_names_in_)
    result = float(model.predict(frame)[0])
    if not np.isfinite(result):
        raise ValueError('The model returned a non-finite prediction.')
    return result
