import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    max_error
)
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Utility
# --------------------------------------------------------------
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def evaluate(y_true, y_pred, n_samples, n_features):
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(r2, n_samples, n_features)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    mx_err = max_error(y_true, y_pred)
    rel_errors = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)

    return {
        "R^2": round(r2, 4),
        "Adjusted R^2": round(adj_r2, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Median AE": round(med_ae, 4),
        "Max Error": round(mx_err, 4),
        "Mean Relative Error (%)": round(np.mean(rel_errors) * 100, 2)
    }


# --------------------------------------------------------------
# Plotting functions
# --------------------------------------------------------------
def save_scatter_plot(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_residual_plot(y_true, y_pred, title, filename):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# --------------------------------------------------------------
# Stratified Train/Test Split + XGBoost
# --------------------------------------------------------------
def summarize_predictions_and_errors(data_path, save_dir):
    df = pd.read_json(data_path)

    # 입력 변수
    X = df[["interval", "count", "degree", "network_size"]]

    # Strata by degree & network_size
    df["bin_degree"] = pd.cut(df["degree"], bins=6, labels=False)
    df["bin_nsize"] = pd.cut(df["network_size"], bins=6, labels=False)

    df["strata_key"] = (
        df["bin_degree"].astype(str) + "_" +
        df["bin_nsize"].astype(str)
    )

    targets = {
        "duplicate": df["duplicate"],
        "reachability": df["reachability"],
        "average_reception_time": df["average_reception_time"]
    }

    results = []

    # stratified split
    X_train, X_test, strata_train, strata_test = train_test_split(
        X, df["strata_key"], test_size=0.2, random_state=42, stratify=df["strata_key"]
    )

    for name, y in targets.items():
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        # 모델 정의
        model = XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            n_jobs=-1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # ----- 시각화 저장 -----
        save_scatter_plot(y_train, y_pred_train,
                          f"{name} - Train Actual vs Predicted",
                          f"{save_dir}/{name}_train.png")

        save_scatter_plot(y_test, y_pred_test,
                          f"{name} - Test Actual vs Predicted",
                          f"{save_dir}/{name}_test.png")

        save_residual_plot(y_train, y_pred_train,
                           f"{name} - Train Residual Plot",
                           f"{save_dir}/{name}_train_residual.png")

        save_residual_plot(y_test, y_pred_test,
                           f"{name} - Test Residual Plot",
                           f"{save_dir}/{name}_test_residual.png")

        # 평가
        train_metrics = evaluate(y_train, y_pred_train, len(X_train), X_train.shape[1])
        test_metrics = evaluate(y_test, y_pred_test, len(X_test), X_test.shape[1])

        results.append({
            "Metric": name,
            "Model": "XGBoost + Stratified Split",
            "Train R^2": train_metrics["R^2"],
            "Test R^2": test_metrics["R^2"],
            "Train MAE": train_metrics["MAE"],
            "Test MAE": test_metrics["MAE"],
            "Train RMSE": train_metrics["RMSE"],
            "Test RMSE": test_metrics["RMSE"],
            "Train MRE (%)": train_metrics["Mean Relative Error (%)"],
            "Test MRE (%)": test_metrics["Mean Relative Error (%)"]
        })

    summary_df = pd.DataFrame(results)
    print(summary_df)
    return summary_df


# --------------------------------------------------------------
# Entry
# --------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "fittit_results.json"
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "fittit_output"
    os.makedirs(save_dir, exist_ok=True)
    summarize_predictions_and_errors(path, save_dir)
