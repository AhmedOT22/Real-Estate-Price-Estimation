# Real Estate Price Estimation

A machine learning web app that trains regression models on real estate data and provides a user-friendly interface for predicting property prices based on user inputs. The app supports model training, prediction, and automatic visualization of results.

![Scikit-learn](https://img.shields.io/badge/framework-scikit--learn-blue)
![Streamlit](https://img.shields.io/badge/ui-streamlit-orange)

---

## Live Demo
Access the deployed app here: [Real Estate Price Estimation App](https://real-estate-price-estimation-utm6wfs9dykdimegfmmwrw.streamlit.app/)

---

## Features
- End-to-end ML pipeline with in-app training, evaluation, and prediction
- Streamlit-based UI for real-time price estimation
- Model selection: Decision Tree, Random Forest, Linear Regression
- Clean form-based input with validation and error handling
- Cached model loading and modular architecture
- Automatic generation of evaluation plots saved to `figures/`

---

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

---

## Model Architecture
- Models: Linear Regression, Decision Tree, Random Forest
- Key components: Scikit-learn pipeline, centralized config
- Evaluation Metric: Mean Absolute Error (MAE)
- Training: Handled via `train_model.py` script with stratified sampling and saved to `models/`
- Visualizations: Model performance plots and tree visualizations saved automatically

---

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

3. **Train the models and generate plots**
   ```bash
   python train_model.py
   ```

4. **Launch the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## Results

| Model               | Train MAE    | Test MAE     |
|---------------------|--------------|--------------|
| Decision Tree       | 52174.76     | 53813.25     |
| Random Forest       | 17291.99     | 42485.02     |
| Linear Regression   | 84268.66     | 86375.82   |

- Random Forest achieves the lowest training error but shows some overfitting on unseen data.
- Decision Tree offers a reasonable balance between simplicity and generalization.

### 📊 Visualizations
Plots automatically generated and saved to `figures/` after training:

- **MAE Comparison Plot:** `figures/mae_comparison.png`
- **Decision Tree Visualization:** `figures/decision_tree.png`
- **Sample Random Forest Tree:** `figures/random_forest_tree.png`

---

## Requirements
- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- pytest (for unit tests)

---

## Author
Developed by [Ahmed Ouazzani](https://github.com/AhmedOT22)

---

## License
MIT License © 2025
