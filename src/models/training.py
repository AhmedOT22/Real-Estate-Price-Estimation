from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import logging

def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target values.

    Returns:
        LinearRegression: Trained linear regression model.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Linear Regression model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to train Linear Regression model: {e}")
        raise

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree Regressor.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target values.

    Returns:
        DecisionTreeRegressor: Trained decision tree model.
    """
    try:
        model = DecisionTreeRegressor(max_depth=5, max_features=10, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Decision Tree model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to train Decision Tree model: {e}")
        raise

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Regressor.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target values.

    Returns:
        RandomForestRegressor: Trained random forest model.
    """
    try:
        model = RandomForestRegressor(n_estimators=200, criterion='absolute_error', random_state=42)
        model.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to train Random Forest model: {e}")
        raise
