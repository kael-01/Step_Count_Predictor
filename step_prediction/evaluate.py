from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lr_model import train_linear_regression
from tree_model import train_tuned_tree


RAW_TRAIN_END_DAY = 60
RAW_TEST_START_DAY = 61
EXPECTED_RAW_DAYS = 90

COLUMN_ALIASES = {
    "steps": ["steps", "step_count"],
    "sleep_minutes": ["sleep_minutes", "sleep_time_minutes"],
    "sleep_hours": ["sleep_hours"],
    "screen_minutes": ["screen_minutes", "screen_time_minutes"],
    "screen_hours": ["screen_hours"],
}


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    df = pd.read_csv(data_path)

    if "day" not in df.columns:
        raise ValueError("Missing required column: day")

    steps_col = find_first_existing_column(df, COLUMN_ALIASES["steps"])
    if steps_col is None:
        raise ValueError(f"Need one of these columns: {COLUMN_ALIASES['steps']}")

    sleep_minutes_col = find_first_existing_column(df, COLUMN_ALIASES["sleep_minutes"])
    sleep_hours_col = find_first_existing_column(df, COLUMN_ALIASES["sleep_hours"])
    if sleep_minutes_col is None and sleep_hours_col is None:
        raise ValueError(
            f"Need one of these columns: {COLUMN_ALIASES['sleep_minutes'] + COLUMN_ALIASES['sleep_hours']}"
        )

    screen_minutes_col = find_first_existing_column(df, COLUMN_ALIASES["screen_minutes"])
    screen_hours_col = find_first_existing_column(df, COLUMN_ALIASES["screen_hours"])
    if screen_minutes_col is None and screen_hours_col is None:
        raise ValueError(
            f"Need one of these columns: {COLUMN_ALIASES['screen_minutes'] + COLUMN_ALIASES['screen_hours']}"
        )

    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    invalid_day_count = int(df["day"].isna().sum())
    if invalid_day_count > 0:
        print(f"Warning: dropped {invalid_day_count} row(s) with invalid day.")
        df = df.dropna(subset=["day"])

    df = df.sort_values("day").copy()

    duplicate_count = int(df.duplicated(subset=["day"]).sum())
    if duplicate_count > 0:
        print(f"Warning: dropped {duplicate_count} duplicate day row(s), kept first.")
        df = df.drop_duplicates(subset=["day"], keep="first")

    df["day"] = df["day"].astype(int)
    df[steps_col] = pd.to_numeric(df[steps_col], errors="coerce")

    if sleep_minutes_col is not None:
        df[sleep_minutes_col] = pd.to_numeric(df[sleep_minutes_col], errors="coerce")
    if sleep_hours_col is not None:
        df[sleep_hours_col] = pd.to_numeric(df[sleep_hours_col], errors="coerce")

    if screen_minutes_col is not None:
        df[screen_minutes_col] = pd.to_numeric(df[screen_minutes_col], errors="coerce")
    if screen_hours_col is not None:
        df[screen_hours_col] = pd.to_numeric(df[screen_hours_col], errors="coerce")

    raw_day_count = len(df)
    if raw_day_count != EXPECTED_RAW_DAYS:
        print(
            f"Warning: expected {EXPECTED_RAW_DAYS} raw day rows, found {raw_day_count}."
        )

    df["steps_t"] = df[steps_col]

    if sleep_minutes_col is not None and sleep_hours_col is not None:
        df["sleep_minutes_t"] = df[sleep_minutes_col].fillna(df[sleep_hours_col] * 60.0)
    elif sleep_minutes_col is not None:
        df["sleep_minutes_t"] = df[sleep_minutes_col]
    else:
        df["sleep_minutes_t"] = df[sleep_hours_col] * 60.0

    if screen_minutes_col is not None and screen_hours_col is not None:
        df["screen_minutes_t"] = df[screen_minutes_col].fillna(df[screen_hours_col] * 60.0)
    elif screen_minutes_col is not None:
        df["screen_minutes_t"] = df[screen_minutes_col]
    else:
        df["screen_minutes_t"] = df[screen_hours_col] * 60.0

    df["day_next"] = df["day"].shift(-1)
    df["steps_next_true"] = df[steps_col].shift(-1)

    if len(df) == 0:
        raise ValueError("No rows available after initial cleaning.")

    df = df.iloc[:-1].copy()

    required_after_shift = [
        "day",
        "day_next",
        "steps_t",
        "sleep_minutes_t",
        "screen_minutes_t",
        "steps_next_true",
    ]
    before_drop_missing = len(df)
    df = df.dropna(subset=required_after_shift).copy()
    dropped_missing_count = before_drop_missing - len(df)
    if dropped_missing_count > 0:
        print(
            f"Warning: dropped {dropped_missing_count} row(s) with missing required fields after target shift."
        )

    df["day_next"] = df["day_next"].astype(int)
    df = df.rename(columns={"day": "day_t"})
    df = df[
        [
            "day_t",
            "day_next",
            "steps_t",
            "sleep_minutes_t",
            "screen_minutes_t",
            "steps_next_true",
        ]
    ].copy()

    return df


def fixed_chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["day_next"] <= RAW_TRAIN_END_DAY].copy()
    test_df = df.loc[df["day_next"] >= RAW_TEST_START_DAY].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Fixed chronological split failed. Train or test is empty.")

    return train_df, test_df


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_value = np.nan if len(y_true) < 2 else r2_score(y_true, y_pred)
    return {"MAE": mae_value, "RMSE": rmse_value, "R2": r2_value}


def make_figures(test_df: pd.DataFrame, output_fig_dir: Path) -> None:
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    y_true = test_df["steps_next_true"]

    min_value = min(
        float(y_true.min()),
        float(test_df["baseline_pred"].min()),
        float(test_df["lr_pred"].min()),
        float(test_df["tree_pred"].min()),
    )
    max_value = max(
        float(y_true.max()),
        float(test_df["baseline_pred"].max()),
        float(test_df["lr_pred"].max()),
        float(test_df["tree_pred"].max()),
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

    axes[0].scatter(y_true, test_df["baseline_pred"], marker="^")
    axes[0].plot([min_value, max_value], [min_value, max_value], "k--")
    axes[0].set_title("Baseline\nPredicted vs Actual")
    axes[0].set_xlabel("Actual Next-Day Steps")
    axes[0].set_ylabel("Predicted Next-Day Steps")

    axes[1].scatter(y_true, test_df["lr_pred"], marker="o")
    axes[1].plot([min_value, max_value], [min_value, max_value], "k--")
    axes[1].set_title("Linear Regression\nPredicted vs Actual")
    axes[1].set_xlabel("Actual Next-Day Steps")

    axes[2].scatter(y_true, test_df["tree_pred"], marker="s")
    axes[2].plot([min_value, max_value], [min_value, max_value], "k--")
    axes[2].set_title("Decision Tree\nPredicted vs Actual")
    axes[2].set_xlabel("Actual Next-Day Steps")

    figure.suptitle("Model Predictions vs Actual (Test Set)")
    figure.tight_layout()
    figure.savefig(output_fig_dir / "pred_vs_actual.png", dpi=150)
    plt.close(figure)

    plt.figure(figsize=(10, 5))
    day_values = pd.to_numeric(test_df["day_next"])

    plt.plot(
        day_values,
        y_true - test_df["lr_pred"],
        marker="o",
        linewidth=1.8,
        label="Linear Regression residual",
    )
    plt.plot(
        day_values,
        y_true - test_df["tree_pred"],
        marker="s",
        linewidth=1.8,
        label="Decision Tree residual",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.6)

    plt.title("Residuals Over Test Days")
    plt.xlabel("Test day")
    plt.ylabel("Residual (actual - predicted)")
    plt.xticks(day_values, rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_fig_dir / "residuals_over_day.png", dpi=150)
    plt.close()


def make_readable_comparison_table(test_df: pd.DataFrame) -> pd.DataFrame:
    readable_df = test_df[
        [
            "day_t",
            "day_next",
            "steps_next_true",
            "baseline_pred",
            "lr_pred",
            "tree_pred",
        ]
    ].copy()
    readable_df = readable_df.rename(
        columns={
            "steps_next_true": "actual_steps_next",
            "baseline_pred": "baseline_predicted_steps",
            "lr_pred": "lr_predicted_steps",
            "tree_pred": "tree_predicted_steps",
        }
    )
    readable_df["baseline_error"] = (
        readable_df["actual_steps_next"] - readable_df["baseline_predicted_steps"]
    )
    readable_df["lr_error"] = (
        readable_df["actual_steps_next"] - readable_df["lr_predicted_steps"]
    )
    readable_df["tree_error"] = (
        readable_df["actual_steps_next"] - readable_df["tree_predicted_steps"]
    )

    numeric_cols = [
        "actual_steps_next",
        "baseline_predicted_steps",
        "lr_predicted_steps",
        "tree_predicted_steps",
        "baseline_error",
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
    train_df, test_df = fixed_chronological_split(prepared_df)

    feature_columns = ["steps_t", "sleep_minutes_t", "screen_minutes_t"]

    X_train = train_df[feature_columns]
    y_train = train_df["steps_next_true"]
    X_test = test_df[feature_columns]
    y_test = test_df["steps_next_true"]

    baseline_pred = test_df["steps_t"].to_numpy()

    lr_model, coefficients_df = train_linear_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    tree_model, best_params, importances_df = train_tuned_tree(X_train, y_train)
    tree_pred = tree_model.predict(X_test)

    predictions_df = test_df[["day_t", "day_next", "steps_t", "steps_next_true"]].copy()
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
            "day_t",
            "day_next",
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

    predictions_df.to_csv(output_dir / "predictions_test.csv", index=False)

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
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    readable_comparison_df = make_readable_comparison_table(predictions_df)
    readable_comparison_df.to_csv(output_dir / "predictions_readable.csv", index=False)

    coefficients_df.to_csv(output_dir / "lr_coefficients.csv", index=False)
    importances_df.to_csv(output_dir / "tree_feature_importances.csv", index=False)

    split_summary_df = pd.DataFrame(
        [
            {
                "raw_days_expected": EXPECTED_RAW_DAYS,
                "raw_train_end_day": RAW_TRAIN_END_DAY,
                "raw_test_start_day": RAW_TEST_START_DAY,
                "prepared_rows": len(prepared_df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_target_day_min": int(train_df["day_next"].min()),
                "train_target_day_max": int(train_df["day_next"].max()),
                "test_target_day_min": int(test_df["day_next"].min()),
                "test_target_day_max": int(test_df["day_next"].max()),
            }
        ]
    )
    split_summary_df.to_csv(output_dir / "split_summary.csv", index=False)

    make_figures(predictions_df, output_fig_dir)

    print(
        f"Rows -> prepared: {len(prepared_df)}, train: {len(train_df)}, test: {len(test_df)}"
    )
    print("Train targets cover day_next = 2..60; test targets cover day_next = 61..90.")
    print("\nLinear Regression coefficients:")
    print(coefficients_df.to_string(index=False))
    print(f"Intercept: {float(lr_model.intercept_):.4f}")
    print("\nDecision Tree best params:")
    print(best_params)
    print("Feature importances:")
    print(importances_df.to_string(index=False))
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nReadable predictions (actual vs baseline vs LR vs Tree):")
    print(readable_comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
