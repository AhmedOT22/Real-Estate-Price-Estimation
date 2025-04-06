from src.data_processing import data_loader
from src.data_processing import features_preparation
from src.models import training
from src.models import evaluation
from src.models import storage

df = data_loader.load_data("data/raw/prices.csv")
X_train, _, y_train, _ = features_preparation.split_data(df)

# Train and save Decision Tree
dt_model = training.train_decision_tree(X_train, y_train)
dt_eval = evaluation.evaluate_model(dt_model, X_train, y_train)
print(f"Decision Tree MAE: {dt_eval:.2f}")
storage.save_model(dt_model, "models/DT_Model.pkl")

# Train and save Random Forest
rf_model = training.train_random_forest(X_train, y_train)
rf_eval = evaluation.evaluate_model(rf_model, X_train, y_train)
print(f"Random Forest MAE: {rf_eval:.2f}")
storage.save_model(rf_model, "models/RF_Model.pkl")

# Train and save Linear Regression
lr_model = training.train_linear_regression(X_train, y_train)
lr_eval = evaluation.evaluate_model(lr_model, X_train, y_train)
print(f"Linear Regression MAE: {lr_eval:.2f}")
storage.save_model(lr_model, "models/LR_Model.pkl")

print(" All models trained and saved successfully.")
