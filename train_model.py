"""Reproducible model comparison; deployment artifacts are never overwritten by default."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_processing.data_loader import load_data
from src.data_processing.features_preparation import split_data
from src.models import storage, training

ROOT = Path(__file__).resolve().parent


def run(input_path=ROOT / 'data/raw/prices.csv', output_dir=ROOT / 'reports/latest', save_models=False):
    input_path, output = Path(input_path), Path(output_dir)
    df = load_data(input_path)
    X_train, X_test, y_train, y_test = split_data(df)
    candidates = {
        'Median baseline': DummyRegressor(strategy='median').fit(X_train, y_train),
        'Decision Tree': training.train_decision_tree(X_train, y_train),
        'Random Forest': training.train_random_forest(X_train, y_train),
        'Linear Regression': training.train_linear_regression(X_train, y_train),
    }
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = pd.DataFrame({'row_index': X_test.index, 'actual_price': y_test.to_numpy()})
    for name, model in candidates.items():
        test_prediction = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        rows.append({'model': name, 'train_MAE': float(train_mae),
                     'test_MAE': float(mean_absolute_error(y_test, test_prediction)),
                     'test_RMSE': float(np.sqrt(mean_squared_error(y_test, test_prediction))),
                     'test_R2': float(r2_score(y_test, test_prediction))})
        predictions[name] = test_prediction
        if save_models:
            storage.save_model(model, output / 'models' / (name.lower().replace(' ', '_') + '.pkl'))
    results = pd.DataFrame(rows).sort_values('test_MAE')
    results.to_csv(output / 'metrics.csv', index=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    metadata = {'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'rows': len(df), 'train_rows': len(X_train), 'test_rows': len(X_test),
                'random_state': 42, 'test_fraction': 0.2, 'stratified_by': 'property_type_Bunglow',
                'split': 'random retrospective holdout; not a future-market backtest',
                'features': list(X_train.columns), 'train_indices': X_train.index.tolist(),
                'test_indices': X_test.index.tolist(), 'python': platform.python_version(),
                'packages': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib']},
                'model_parameters': {name: model.get_params() for name, model in candidates.items()}}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(results['model'], results['test_MAE'], color='#237a80')
    ax.invert_yaxis()
    ax.set(xlabel='Holdout MAE (dataset price units)', title='Retrospective model comparison')
    fig.tight_layout()
    fig.savefig(output / 'mae_comparison.png', dpi=150)
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/prices.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/latest')
    parser.add_argument('--save-models', action='store_true', help='Save candidates beneath OUTPUT/models; does not replace app models')
    args = parser.parse_args()
    print(run(args.input, args.output, args.save_models).to_string(index=False))


if __name__ == '__main__':
    main()
