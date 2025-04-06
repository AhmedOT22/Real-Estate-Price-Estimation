# Real Estate Price Estimation

A machine learning web app that trains regression models on real estate data and provides a user-friendly interface for predicting property prices based on user inputs.

![Scikit-learn](https://img.shields.io/badge/framework-scikit--learn-blue)
![Streamlit](https://img.shields.io/badge/ui-streamlit-orange)
![Model Accuracy](https://img.shields.io/badge/RandomForrest-MAE%20~%2011k-success)


## Live Demo
Access the deployed app here: [Real Estate Price Estimation App](https://real-estate-price-estimation-utm6wfs9dykdimegfmmwrw.streamlit.app/)


## Features
- End-to-end ML pipeline with in-app training and prediction
- Streamlit-based UI for real-time price estimation
- Model selection: Decision Tree, Random Forest, Linear Regression
- Clean form-based input with validation
- Cached model loading and modular architecture


## Dataset
- Path: `data/raw/prices.csv`
- Target Column: `price`
- Features Used:
  - `year_sold`, `property_tax`, `insurance`, `beds`, `baths`, `sqft`, `year_built`, `lot_size`
  - Boolean flags: `basement`, `popular`, `recession`
  - Derived: `property_age`
  - One-hot encoded: `property_type_Bunglow`, `property_type_Condo`


## Model Architecture
- Models: Linear Regression, Decision Tree, Random Forest
- Key components: Scikit-learn pipeline, centralized config
- Evaluation Metric: Mean Absolute Error (MAE)
- Training: Handled in-app or via `train_model.py` script with stratified sampling


## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AhmedOT22/loan-eligibility-app.git
   cd loan-eligibility-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```


## Results
- Evaluation is performed on the training set immediately after each model is trained.
- Evaluation metrics (Mean Absolute Error):
  - Decision Tree MAE: **50,987.39**
  - Random Forest MAE: **17,095.51**
  - Linear Regression MAE: **86,913.32**
- Clean input validation (e.g., no future year entries)
- Dynamic model selection from sidebar


## Requirements
- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib


## Author
Developed by [Ahmed Ouazzani](https://github.com/AhmedOT22)

## License
MIT License © 2025
