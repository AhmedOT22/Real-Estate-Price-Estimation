import pandas as pd
import logging

def predict_single(model, input_values):
    """
    Predicts a single output using the given model and input values.

    Args:
        model: Trained ML model with feature_names_in_ attribute.
        input_values (list): List of feature values matching the model's input order.

    Returns:
        float: Predicted value.
    """
    try:
        # Automatically correct shape if input_values is flat
        if isinstance(input_values[0], (int, float)):
            input_values = [input_values]  # wrap into a list of lists

        input_df = pd.DataFrame(input_values, columns=model.feature_names_in_)
        prediction = model.predict(input_df)
        logging.info("Prediction completed successfully.")
        return prediction[0]
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
