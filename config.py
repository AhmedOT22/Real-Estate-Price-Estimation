# config.py

MODEL_PATHS = {
    "Decision Tree": "models/DT_Model.pkl",
    "Random Forest": "models/RF_Model.pkl",
    "Linear Regression": "models/LR_Model.pkl"
}

# Input defaults and boundaries
INPUT_RANGES = {
    "year_built_min": 1800,
    "year_sold_max": 2050,
    "property_tax_max": 100000,
    "insurance_max": 50000,
    "sqft_min": 100,
    "sqft_max": 20000,
    "lot_size_min": 100,
    "lot_size_max": 100000
}
