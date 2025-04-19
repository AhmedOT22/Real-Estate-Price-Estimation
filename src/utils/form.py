import streamlit as st
from datetime import datetime
from src.config import INPUT_RANGES

def fetch_input(feature_names):
    """
    Collects user input via Streamlit widgets to generate feature values for prediction.

    Args:
        feature_names (list): List of feature names expected by the model.

    Returns:
        list: Ordered list of input values matching the feature names.
    """
    current_year = datetime.now().year

    # Property type selection (one-hot encoding)
    property_type = st.radio("Property Type", ["Bunglow", "Condo"], help="Select the type of property")
    property_type_bunglow = int(property_type == "Bunglow")
    property_type_condo = int(property_type == "Condo")

    # Year inputs with validation
    year_built = st.number_input("Year Built", min_value=INPUT_RANGES["year_built_min"], max_value=current_year, value=2000, help="Construction year of the property")
    year_sold = st.number_input("Year Sold", min_value=year_built, max_value=current_year, value=current_year, help="Year the property was sold")
    calculated_age = year_sold - year_built

    # Numeric financial and physical attributes
    property_tax = st.number_input("Annual Property Tax ($)", min_value=0, max_value=INPUT_RANGES["property_tax_max"], value=3000, help="Yearly tax paid for the property")
    insurance = st.number_input("Annual Insurance Cost ($)", min_value=0, max_value=INPUT_RANGES["insurance_max"], value=1000, help="Annual insurance premium")
    sqft = st.number_input("Total Living Area (sqft)", min_value=INPUT_RANGES["sqft_min"], max_value=INPUT_RANGES["sqft_max"], value=1500, help="Size of the home")
    lot_size = st.number_input("Lot Size (sqft)", min_value=INPUT_RANGES["lot_size_min"], max_value=INPUT_RANGES["lot_size_max"], value=5000, help="Total size of the land")

    # Categorical/boolean fields
    feature_inputs = {
        'year_sold': year_sold,
        'property_tax': property_tax,
        'insurance': insurance,
        'beds': st.slider("Number of Bedrooms", 1, 10, 3, help="Total bedrooms in the property"),
        'baths': st.slider("Number of Bathrooms", 1, 10, 2, help="Total bathrooms in the property"),
        'sqft': sqft,
        'year_built': year_built,
        'lot_size': lot_size,
        'basement': int(st.checkbox("Includes Basement", value=False, help="Check if there's a basement")),
        'popular': int(st.checkbox("Featured Listing", value=False, help="Check if the property is listed as popular")),
        'recession': int(st.checkbox("Sold During Recession", value=False, help="Check if property was sold during a recession year")),
        'property_age': calculated_age,
        'property_type_Bunglow': property_type_bunglow,
        'property_type_Condo': property_type_condo
    }

    # Return values in the correct order as expected by the model
    return [feature_inputs.get(feat, 0) for feat in feature_names]