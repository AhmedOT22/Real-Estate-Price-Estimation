import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os
import logging

def save_decision_tree(model, feature_names, save_path="figures/decision_tree.png"):
    """
    Saves a visualization of a Decision Tree model.

    Args:
        model: Trained DecisionTreeRegressor
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, filled=True)
        plt.title("Decision Tree Visualization")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Decision Tree visualization saved to {save_path}.")
    except Exception as e:
        logging.error(f"Failed to save Decision Tree plot: {e}")
        raise

def save_random_forest_tree(model, feature_names, save_path="figures/random_forest_tree.png"):
    """
    Saves a visualization of one Decision Tree from a Random Forest model.

    Args:
        model: Trained RandomForestRegressor
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(20, 10))
        plot_tree(model.estimators_[0], feature_names=feature_names, filled=True)
        plt.title("Sample Tree from Random Forest")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Random Forest sample tree saved to {save_path}.")
    except Exception as e:
        logging.error(f"Failed to save Random Forest tree plot: {e}")
        raise

def save_mae_comparison(models, train_maes, test_maes, save_path="figures/mae_comparison.png"):
    """
    Saves a bar chart comparing MAE values.

    Args:
        models: List of model names
        train_maes: List of training MAEs
        test_maes: List of testing MAEs
        save_path: Path to save the plot
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        x = range(len(models))
        plt.figure(figsize=(10, 6))
        plt.bar(x, train_maes, width=0.4, label='Train MAE', align='center')
        plt.bar([i + 0.4 for i in x], test_maes, width=0.4, label='Test MAE', align='center')
        plt.xlabel('Model')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Model Performance Comparison')
        plt.xticks([i + 0.2 for i in x], models)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"MAE comparison plot saved to {save_path}.")
    except Exception as e:
        logging.error(f"Failed to save MAE comparison plot: {e}")
        raise
