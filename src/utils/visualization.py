import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import logging

def visualize_tree(model, feature_names):
    """
    Plots a decision tree using matplotlib.

    Args:
        model: Trained DecisionTreeRegressor or tree from RandomForest.
        feature_names (list): List of feature names used in training.
    """
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, filled=True)
        plt.title("Decision Tree Visualization")
        plt.show()
        logging.info("Tree visualization displayed successfully.")
    except Exception as e:
        logging.error(f"Failed to plot decision tree: {e}")
        raise