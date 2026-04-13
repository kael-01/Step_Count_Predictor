from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LinearRegression, pd.DataFrame]:
    """
    Train a linear regression model and return model + coefficients table.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    coefficients_df = pd.DataFrame(
        {"feature": X_train.columns, "coefficient": model.coef_}
    )

    return model, coefficients_df
