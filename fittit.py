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
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
# Heatmap (Interval × Count) — multiple degrees × nsizes
# --------------------------------------------------------------
def save_heatmap(model, df, save_path, metric_name, fixed_degrees, fixed_nsizes):
    interval_vals = np.linspace(df["interval"].min(), df["interval"].max(), 30)
    count_vals = np.linspace(df["count"].min(), df["count"].max(), 30)

    I, C = np.meshgrid(interval_vals, count_vals)

    rows = len(fixed_degrees)
    cols = len(fixed_nsizes)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, degree_val in enumerate(fixed_degrees):
        for j, nsize_val in enumerate(fixed_nsizes):

            grid = pd.DataFrame({
                "interval": I.ravel(),
                "count": C.ravel(),
                "degree": degree_val,
                "network_size": nsize_val
            })

            Z = model.predict(grid).reshape(I.shape)

            ax = axes[i][j]
            sns.heatmap(
                Z,
                xticklabels=False,
                yticklabels=False,
                cmap="viridis",
                ax=ax
            )
            ax.set_title(f"{metric_name}\nDeg={degree_val}, Nsize={nsize_val}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------------------------------------------
# Partial Dependence Plot (PDP)
# --------------------------------------------------------------
def save_pdp(model, df, save_path, target_name, var_name, fixed_values):
    x_vals = np.linspace(df[var_name].min(), df[var_name].max(), 100)

    input_df = pd.DataFrame({
        "interval": fixed_values["interval"],
        "count": fixed_values["count"],
        "degree": fixed_values["degree"],
        "network_size": fixed_values["network_size"]
    }, index=range(len(x_vals)))

    input_df[var_name] = x_vals

    y_pred = model.predict(input_df)

    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_pred)
    plt.title(f"PDP: {target_name} vs {var_name}")
    plt.xlabel(var_name)
    plt.ylabel(target_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------------------------------------------
# Density Scatter Plot
# --------------------------------------------------------------
def save_density_scatter(y_true, y_pred, save_path, title):
    plt.figure(figsize=(7, 6))
    sns.kdeplot(x=y_true, y=y_pred, fill=True, cmap="viridis", thresh=0.05)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------------------------------------------
# 3D Surface Plot
# --------------------------------------------------------------
def save_3d_surface(model, df, save_path, metric_name, fixed_degree, fixed_nsize):
    interval_vals = np.linspace(df["interval"].min(), df["interval"].max(), 40)
    count_vals = np.linspace(df["count"].min(), df["count"].max(), 40)

    I, C = np.meshgrid(interval_vals, count_vals)

    grid = pd.DataFrame({
        "interval": I.ravel(),
        "count": C.ravel(),
        "degree": fixed_degree,
        "network_size": fixed_nsize
    })

    Z = model.predict(grid).reshape(I.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(I, C, Z, cmap="viridis", edgecolor='none')

    ax.set_title(f"3D Surface: {metric_name}\nDeg={fixed_degree}, Nsize={fixed_nsize}")
    ax.set_xlabel("Interval")
    ax.set_ylabel("Count")
    ax.set_zlabel(metric_name)

    fig.colorbar(surf, shrink=0.6, aspect=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------------------------------------------
# Stratified Train/Test Split + XGBoost + Visualization
# --------------------------------------------------------------
def summarize_predictions_and_errors(data_path, save_dir):
    df = pd.read_json(data_path)

    X = df[["interval", "count", "degree", "network_size"]]

    # Strata
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

    X_train, X_test, strata_train, strata_test = train_test_split(
        X, df["strata_key"], test_size=0.2, random_state=42, stratify=df["strata_key"]
    )

    # graph settings
    fixed_degrees = [df["degree"].min(), df["degree"].median(), df["degree"].max()]
    fixed_nsizes = [df["network_size"].min(), df["network_size"].median(), df["network_size"].max()]

    for name, y in targets.items():
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]

        model = XGBRegressor(
            n_estimators=600,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Heatmap
        save_heatmap(
            model, df,
            f"{save_dir}/{name}_heatmap.png",
            name, fixed_degrees, fixed_nsizes
        )

        # PDP (4 variables)
        for var in ["interval", "count", "degree", "network_size"]:
            fixed_values = {
                "interval": df["interval"].median(),
                "count": df["count"].median(),
                "degree": df["degree"].median(),
                "network_size": df["network_size"].median()
            }
            save_pdp(
                model, df,
                f"{save_dir}/{name}_pdp_{var}.png",
                name, var, fixed_values
            )

        # Feature Importance
        importance = model.feature_importances_
        plt.figure(figsize=(7, 5))
        sns.barplot(x=importance, y=X.columns)
        plt.title(f"Feature Importance: {name}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name}_feature_importance.png")
        plt.close()

        # Density Scatter
        save_density_scatter(
            y_test, y_pred_test,
            f"{save_dir}/{name}_density.png",
            f"{name} - Density Scatter"
        )

        # 3D Surface Plot
        for fixed_degree in fixed_degrees:
            for fixed_nsize in fixed_nsizes:
                save_3d_surface(model, df, f"{save_dir}/{name}_3d_deg{fixed_degree}_n{fixed_nsize}.png",
                        name, fixed_degree, fixed_nsize)

        # metrics
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


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "fittit_results.json"
    save_dir = sys.argv[2] if len(sys.argv) > 2 else "fittit_output"
    os.makedirs(save_dir, exist_ok=True)
    summarize_predictions_and_errors(path, save_dir)
