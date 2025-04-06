import streamlit as st
import pandas as pd
import logging
import numpy as np
from datetime import datetime
from src.models import prediction
from src.models import storage
from src.utils.form import fetch_input
from config import MODEL_PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Real Estate Price Predictor", layout="centered")
st.title("Real Estate Price Prediction App")
st.markdown("Enter property details below to estimate its market price.")

# Load all models from config paths
@st.cache_resource
def load_models():
    """
    Loads all pre-trained models specified in the MODEL_PATHS config.

    Returns:
        dict: Dictionary of model name to model object
    """
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = storage.load_model(path)
            logging.info(f"Loaded model: {name} from {path}")
        except Exception as e:
            logging.error(f"Failed to load model '{name}' from {path}: {e}")
    return models

# Load models once
model_dict = load_models()

if not model_dict:
    st.error("No models found. Please ensure models are available in the 'models/' directory.")
    st.stop()  # Exit if no models were loaded

model_choice = st.sidebar.selectbox("Choose Model", list(model_dict.keys()))

if model_choice is None:
    st.warning("Please select a model to proceed.")
    st.stop()

model = model_dict[model_choice]

feature_names = model.feature_names_in_

# Centered layout for form
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.header("Property Details")
    input_values = fetch_input(feature_names)

    if st.button("Predict Price"):
        try:
            predicted_price = prediction.predict_single(model, [input_values])
            st.success(f"Estimated Property Price: ${predicted_price:,.2f}")
            logging.info(f"Prediction successful: ${predicted_price:,.2f} using {model_choice}")
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            st.error(f"Prediction failed: {e}")
