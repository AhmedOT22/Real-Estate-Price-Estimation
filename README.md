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
   git clone https://github.com/AhmedOT22/Real-Estate-Price-Estimation.git
   cd Real-Estate-Price-Estimation
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

| Model               | Train MAE    | Test MAE     |
|---------------------|--------------|--------------|
| Decision Tree       | 49,856.81    | 53,343.82    |
| Random Forest       | 16,970.25    | 46,377.51    |
| Linear Regression   | 87,275.74    | 85,566.09    |

- Random Forest achieves the lowest training error but shows some overfitting on unseen data.
- Decision Tree offers a reasonable balance between simplicity and generalization.

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