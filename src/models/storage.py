import pickle
import logging

def save_model(model, filename):
    """
    Saves the given model to disk using pickle.

    Args:
        model: Trained machine learning model.
        filename (str): Path where the model should be saved.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save model to {filename}: {e}")
        raise

def load_model(filename):
    """
    Loads a machine learning model from disk using pickle.

    Args:
        filename (str): Path to the saved model file.

    Returns:
        The loaded model.
    """
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {filename}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {filename}: {e}")
        raise
