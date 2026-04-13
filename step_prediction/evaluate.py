from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lr_model import train_linear_regression
from tree_model import train_tuned_tree


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    df = pd.read_csv(data_path)

    for required_column in ["date", "steps"]:
        if required_column not in df.columns:
            raise ValueError(f"Missing required column: {required_column}")

    has_sleep_hours = "sleep_hours" in df.columns
    has_sleep_minutes = "sleep_minutes" in df.columns
    if not (has_sleep_hours or has_sleep_minutes):
        raise ValueError("Need one of these columns: sleep_hours or sleep_minutes")

    has_screen_minutes = "screen_minutes" in df.columns
    has_screen_hours = "screen_hours" in df.columns
    if not (has_screen_minutes or has_screen_hours):
        raise ValueError("Need one of these columns: screen_minutes or screen_hours")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_count = int(df["date"].isna().sum())
    if invalid_date_count > 0:
        print(f"Warning: dropped {invalid_date_count} row(s) with invalid date.")
        df = df.dropna(subset=["date"])

    df = df.sort_values("date").copy()

    duplicate_count = int(df.duplicated(subset=["date"]).sum())
    if duplicate_count > 0:
        print(f"Warning: dropped {duplicate_count} duplicate date row(s), kept first.")
        df = df.drop_duplicates(subset=["date"], keep="first")

    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")

    if has_sleep_hours:
        df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors="coerce")
    if has_sleep_minutes:
        df["sleep_minutes"] = pd.to_numeric(df["sleep_minutes"], errors="coerce")
    if has_screen_minutes:
        df["screen_minutes"] = pd.to_numeric(df["screen_minutes"], errors="coerce")
    if has_screen_hours:
        df["screen_hours"] = pd.to_numeric(df["screen_hours"], errors="coerce")

    df["steps_t"] = df["steps"]

    if has_sleep_hours and has_sleep_minutes:
        df["sleep_t"] = df["sleep_hours"].fillna(df["sleep_minutes"] / 60.0)
    elif has_sleep_hours:
        df["sleep_t"] = df["sleep_hours"]
    else:
        df["sleep_t"] = df["sleep_minutes"] / 60.0

    if has_screen_minutes and has_screen_hours:
        df["screen_t"] = df["screen_minutes"].fillna(df["screen_hours"] * 60.0)
    elif has_screen_minutes:
        df["screen_t"] = df["screen_minutes"]
    else:
        df["screen_t"] = df["screen_hours"] * 60.0

    df["date_next"] = df["date"].shift(-1)
    df["steps_next_true"] = df["steps"].shift(-1)

    if len(df) == 0:
        raise ValueError("No rows available after initial cleaning.")

    df = df.iloc[:-1].copy()

    required_after_shift = [
        "date",
        "date_next",
        "steps_t",
        "sleep_t",
        "screen_t",
        "steps_next_true",
    ]
    before_drop_missing = len(df)
    df = df.dropna(subset=required_after_shift).copy()
    dropped_missing_count = before_drop_missing - len(df)
    if dropped_missing_count > 0:
        print(
            f"Warning: dropped {dropped_missing_count} row(s) with missing required fields after target shift."
        )

    df = df.rename(columns={"date": "date_t"})
    df = df[
        ["date_t", "date_next", "steps_t", "sleep_t", "screen_t", "steps_next_true"]
    ].copy()

    return df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("Prepared data is too small. Need at least 2 rows.")

    test_size = max(1, round(0.2 * n_rows))
    if test_size >= n_rows:
        test_size = 1

    split_index = n_rows - test_size
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Chronological split failed. Train or test is empty.")

    return train_df, test_df


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))

    if len(y_true) < 2:
        r2_value = np.nan
    else:
        r2_value = r2_score(y_true, y_pred)

    return {"MAE": mae_value, "RMSE": rmse_value, "R2": r2_value}


def make_figures(test_df: pd.DataFrame, output_fig_dir: Path) -> None:
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    y_true = test_df["steps_next_true"]

    min_value = min(
        float(y_true.min()),
        float(test_df["lr_pred"].min()),
        float(test_df["tree_pred"].min()),
    )
    max_value = max(
        float(y_true.max()),
        float(test_df["lr_pred"].max()),
        float(test_df["tree_pred"].max()),
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].scatter(y_true, test_df["lr_pred"], color="#1f77b4", marker="o")
    axes[0].plot([min_value, max_value], [min_value, max_value], "k--")
    axes[0].set_title("Linear Regression\nPredicted vs Actual")
    axes[0].set_xlabel("Actual Next-Day Steps")
    axes[0].set_ylabel("Predicted Next-Day Steps")

    axes[1].scatter(y_true, test_df["tree_pred"], color="#ff7f0e", marker="s")
    axes[1].plot([min_value, max_value], [min_value, max_value], "k--")
    axes[1].set_title("Decision Tree\nPredicted vs Actual")
    axes[1].set_xlabel("Actual Next-Day Steps")

    figure.suptitle("Model Predictions vs Actual (Test Set)")
    figure.tight_layout()
    figure.savefig(output_fig_dir / "pred_vs_actual.png", dpi=150)
    plt.close(figure)

    plt.figure(figsize=(9, 5))
    date_values = pd.to_datetime(test_df["date_next"])
    plt.plot(
        date_values,
        y_true - test_df["lr_pred"],
        marker="o",
        label="LR residual",
    )
    plt.plot(
        date_values,
        y_true - test_df["tree_pred"],
        marker="s",
        label="Tree residual",
    )
    plt.axhline(0.0, linestyle="--")

    plt.title("Residuals Over Time")
    plt.xlabel("date_next")
    plt.ylabel("Residual (y_true - pred)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_fig_dir / "residuals_over_time.png", dpi=150)
    plt.close()


def make_readable_comparison_table(test_df: pd.DataFrame) -> pd.DataFrame:
    readable_df = test_df[
        ["date_t", "date_next", "steps_next_true", "lr_pred", "tree_pred"]
    ].copy()
    readable_df = readable_df.rename(
        columns={
            "steps_next_true": "actual_steps_next",
            "lr_pred": "lr_predicted_steps",
            "tree_pred": "tree_predicted_steps",
        }
    )
    readable_df["lr_error"] = (
        readable_df["actual_steps_next"] - readable_df["lr_predicted_steps"]
    )
    readable_df["tree_error"] = (
        readable_df["actual_steps_next"] - readable_df["tree_predicted_steps"]
    )

    numeric_cols = [
        "actual_steps_next",
        "lr_predicted_steps",
        "tree_predicted_steps",
        "lr_error",
        "tree_error",
    ]
    readable_df[numeric_cols] = readable_df[numeric_cols].round(2)
    return readable_df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data.csv"
    output_dir = base_dir / "outputs"
    output_fig_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    prepared_df = load_and_prepare_data(data_path)
    train_df, test_df = chronological_split(prepared_df)

    feature_columns = ["steps_t", "sleep_t", "screen_t"]

    X_train = train_df[feature_columns]
    y_train = train_df["steps_next_true"]
    X_test = test_df[feature_columns]
    y_test = test_df["steps_next_true"]

    baseline_pred = test_df["steps_t"].to_numpy()

    lr_model, coefficients_df = train_linear_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    tree_model, best_params, importances_df = train_tuned_tree(X_train, y_train)
    tree_pred = tree_model.predict(X_test)

    predictions_df = test_df[["date_t", "date_next", "steps_t", "steps_next_true"]].copy()
    predictions_df["baseline_pred"] = baseline_pred
    predictions_df["lr_pred"] = lr_pred
    predictions_df["tree_pred"] = tree_pred

    predictions_df["baseline_abs_err"] = (
        predictions_df["steps_next_true"] - predictions_df["baseline_pred"]
    ).abs()
    predictions_df["lr_abs_err"] = (
        predictions_df["steps_next_true"] - predictions_df["lr_pred"]
    ).abs()
    predictions_df["tree_abs_err"] = (
        predictions_df["steps_next_true"] - predictions_df["tree_pred"]
    ).abs()

    predictions_df["baseline_sq_err"] = (
        predictions_df["steps_next_true"] - predictions_df["baseline_pred"]
    ) ** 2
    predictions_df["lr_sq_err"] = (
        predictions_df["steps_next_true"] - predictions_df["lr_pred"]
    ) ** 2
    predictions_df["tree_sq_err"] = (
        predictions_df["steps_next_true"] - predictions_df["tree_pred"]
    ) ** 2

    predictions_df = predictions_df[
        [
            "date_t",
            "date_next",
            "steps_t",
            "steps_next_true",
            "baseline_pred",
            "lr_pred",
            "tree_pred",
            "baseline_abs_err",
            "lr_abs_err",
            "tree_abs_err",
            "baseline_sq_err",
            "lr_sq_err",
            "tree_sq_err",
        ]
    ].copy()

    predictions_path = output_dir / "predictions_test.csv"
    predictions_df.to_csv(predictions_path, index=False)

    metrics_rows = []
    for model_name, prediction_column in [
        ("baseline", "baseline_pred"),
        ("lr", "lr_pred"),
        ("tree", "tree_pred"),
    ]:
        metric_values = compute_metrics(
            y_true=y_test,
            y_pred=predictions_df[prediction_column].to_numpy(),
        )
        metrics_rows.append(
            {
                "model": model_name,
                "MAE": metric_values["MAE"],
                "RMSE": metric_values["RMSE"],
                "R2": metric_values["R2"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows, columns=["model", "MAE", "RMSE", "R2"])
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    readable_comparison_df = make_readable_comparison_table(predictions_df)
    readable_comparison_path = output_dir / "predictions_readable.csv"
    readable_comparison_df.to_csv(readable_comparison_path, index=False)

    make_figures(predictions_df, output_fig_dir)

    print(
        f"Rows -> total: {len(prepared_df)}, train: {len(train_df)}, test: {len(test_df)}"
    )
    print("\nLinear Regression coefficients:")
    print(coefficients_df.to_string(index=False))
    print(f"Intercept: {float(lr_model.intercept_):.4f}")

    print("\nDecision Tree best params:")
    print(best_params)
    print("Feature importances:")
    print(importances_df.to_string(index=False))

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nReadable predictions (actual vs LR vs Tree):")
    print(readable_comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
