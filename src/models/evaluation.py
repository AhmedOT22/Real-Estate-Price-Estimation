from sklearn.metrics import mean_absolute_error
import logging

def evaluate_model(model, X, y, label="Model"):
    """
    Evaluates a model's prediction performance using Mean Absolute Error.

    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Feature input for prediction.
        y (pd.Series): True target values.
        label (str): Label to identify the model in logs.

    Returns:
        float: Mean Absolute Error
    """
    try:
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        logging.info(f"{label} MAE: {mae:.2f}")
        return mae
    except Exception as e:
        logging.error(f"Evaluation failed for {label}: {e}")
        raise
