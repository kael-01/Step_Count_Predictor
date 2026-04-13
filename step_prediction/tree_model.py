from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


def train_tuned_tree(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[DecisionTreeRegressor, dict, pd.DataFrame]:
    """
    Tune a small decision tree using a chronological inner validation split.
    """
    n_train = len(X_train)
    if n_train < 2:
        raise ValueError("Need at least 2 training rows for decision tree tuning.")

    split_index = int(n_train * 0.8)
    if split_index < 1:
        split_index = 1
    if split_index >= n_train:
        split_index = n_train - 1

    X_core = X_train.iloc[:split_index]
    y_core = y_train.iloc[:split_index]
    X_val = X_train.iloc[split_index:]
    y_val = y_train.iloc[split_index:]

    depth_values = [2, 3, 4, 5, 6]
    leaf_values = [1, 2, 5]

    best_params = None
    best_val_mae = float("inf")

    for max_depth in depth_values:
        for min_samples_leaf in leaf_values:
            model = DecisionTreeRegressor(
                random_state=42,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )

            try:
                model.fit(X_core, y_core)
                val_pred = model.predict(X_val)
                val_mae = mean_absolute_error(y_val, val_pred)
            except ValueError:
                continue

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }

    if best_params is None:
        best_params = {"max_depth": 2, "min_samples_leaf": 1}

    final_model = DecisionTreeRegressor(random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    importances_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": final_model.feature_importances_}
    )

    return final_model, best_params, importances_df
