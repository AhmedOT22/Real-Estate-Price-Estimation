# Real Estate Price Estimation

A machine learning web app that trains regression models on real estate data and provides a user-friendly interface for predicting property prices based on user inputs.

![Scikit-learn](https://img.shields.io/badge/framework-scikit--learn-blue)
![Streamlit](https://img.shields.io/badge/ui-streamlit-orange)
![Model Accuracy](https://img.shields.io/badge/model-MAE%20~%2011k-success)


## How To Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train and save models
python train_model.py

# Launch Streamlit app
streamlit run app.py
```


## Dataset
- Path: `data/raw/prices.csv`
- Target Column: `price`
- Features Used:
  - `year_sold`, `property_tax`, `insurance`, `beds`, `baths`, `sqft`, `year_built`, `lot_size`
  - Boolean flags: `basement`, `popular`, `recession`
  - Derived: `property_age`
  - One-hot encoded: `property_type_Bunglow`, `property_type_Condo`
- Preprocessing:
  - One-hot encoding for `property_type`
  - Derived feature: `property_age = year_sold - year_built`
  - Stratified train-test split based on `property_type_Bunglow`


## Model Architecture
- Models: Linear Regression, Decision Tree, Random Forest
- Key components: Scikit-learn pipeline, centralized config
- Evaluation Metric: Mean Absolute Error (MAE)
- Training: Handled in-app or via `train_model.py` script with stratified sampling


## Results
- Evaluation is performed on the training set immediately after each model is trained.
- Evaluation metrics (Mean Absolute Error):
  - Decision Tree MAE: 50,987.39
  - Random Forest MAE: 17,095.51 (Best Performer)
  - Linear Regression MAE: 86,913.32


## 📋 Requirements
- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib


## License
MIT License © 2024
