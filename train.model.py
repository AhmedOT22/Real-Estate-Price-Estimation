from src.data_processing import data_loader
from src.data_processing import features_preparation
from src.models import training
from src.models import evaluation
from src.models import storage
from src.models import visualization

# Load data
df = data_loader.load_data("data/raw/prices.csv")
X_train, X_test, y_train, y_test = features_preparation.split_data(df)

# Train and save Decision Tree
dt_model = training.train_decision_tree(X_train, y_train)
dt_train_mae = evaluation.evaluate_model(dt_model, X_train, y_train)
dt_test_mae = evaluation.evaluate_model(dt_model, X_test, y_test)
print(f"Decision Tree - Train MAE: {dt_train_mae:.2f} | Test MAE: {dt_test_mae:.2f}")
storage.save_model(dt_model, "models/DT_Model.pkl")

# Train and save Random Forest
rf_model = training.train_random_forest(X_train, y_train)
rf_train_mae = evaluation.evaluate_model(rf_model, X_train, y_train)
rf_test_mae = evaluation.evaluate_model(rf_model, X_test, y_test)
print(f"Random Forest - Train MAE: {rf_train_mae:.2f} | Test MAE: {rf_test_mae:.2f}")
storage.save_model(rf_model, "models/RF_Model.pkl")

# Train and save Linear Regression
lr_model = training.train_linear_regression(X_train, y_train)
lr_train_mae = evaluation.evaluate_model(lr_model, X_train, y_train)
lr_test_mae = evaluation.evaluate_model(lr_model, X_test, y_test)
print(f"Linear Regression - Train MAE: {lr_train_mae:.2f} | Test MAE: {lr_test_mae:.2f}")
storage.save_model(lr_model, "models/LR_Model.pkl")

print("All models trained, evaluated, and saved successfully.")

# Save visualizations
visualization.save_decision_tree(dt_model, X_train.columns)
visualization.save_random_forest_tree(rf_model, X_train.columns)
visualization.save_mae_comparison(
    models=["Decision Tree", "Random Forest", "Linear Regression"],
    train_maes=[dt_train_mae, rf_train_mae, lr_train_mae],
    test_maes=[dt_test_mae, rf_test_mae, lr_test_mae]
)
