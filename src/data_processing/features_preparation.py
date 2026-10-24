"""Validate the prepared feature table and make a repeatable holdout split."""
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES = ('year_sold', 'property_tax', 'insurance', 'beds', 'baths', 'sqft',
            'year_built', 'lot_size', 'basement', 'popular', 'recession',
            'property_age', 'property_type_Bunglow', 'property_type_Condo')


def split_data(df, target='price', stratify_col='property_type_Bunglow', random_state=42):
    expected = set(FEATURES) | {target}
    if set(df.columns) != expected or len(df.columns) != len(expected):
        raise ValueError(f'Expected target {target!r} and exactly these features: {FEATURES}')
    try:
        values = df.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Prepared features and target must be numeric.') from error
    if not np.isfinite(values).all():
        raise ValueError('Prepared features and target must be finite, with no missing values.')
    if (values < 0).any():
        raise ValueError('Prepared features and prices must be non-negative.')
    if not df['property_age'].eq(df['year_sold'] - df['year_built']).all():
        raise ValueError('property_age must equal year_sold - year_built.')
    flags = ['basement', 'popular', 'recession', 'property_type_Bunglow', 'property_type_Condo']
    if not df[flags].isin([0, 1]).all().all():
        raise ValueError('Boolean and one-hot features must be zero or one.')
    if not (df['property_type_Bunglow'] + df['property_type_Condo']).eq(1).all():
        raise ValueError('Exactly one property type must be selected.')
    X, y = df.loc[:, list(FEATURES)], df[target]
    return train_test_split(X, y, test_size=0.2, stratify=X[stratify_col], random_state=random_state)
