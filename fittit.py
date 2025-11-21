import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    max_error
)

def summarize_predictions_and_errors(data_path):
    # 데이터 로드
    df = pd.read_json(data_path)

    # 입력 변수 설정
    X = df[["interval", "count", "degree", "network_size"]]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    n_samples = X_poly.shape[0]
    n_features = X_poly.shape[1]

    # 타겟 변수
    y_dup = df["duplicate"]
    y_reach = df["reachability"]
    y_time = df["average_reception_time"]

    # 회귀 모델 학습
    model_dup = LinearRegression().fit(X_poly, y_dup)
    model_reach = LinearRegression().fit(X_poly, y_reach)
    model_time = LinearRegression().fit(X_poly, y_time)

    # 예측
    preds_dup = model_dup.predict(X_poly)
    preds_reach = model_reach.predict(X_poly)
    preds_time = model_time.predict(X_poly)

    # 지표 계산 함수
    def adjusted_r2(r2, n, p):
        """Adjusted R^2 계산"""
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # 결과 요약
    results = []
    for name, y_true, y_pred in [
        ("duplicate", y_dup, preds_dup),
        ("reachability", y_reach, preds_reach),
        ("average_reception_time", y_time, preds_time)
    ]:
        r2 = r2_score(y_true, y_pred)
        adj_r2 = adjusted_r2(r2, n_samples, n_features)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        med_ae = median_absolute_error(y_true, y_pred)
        max_err = max_error(y_true, y_pred)

        rel_errors = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)

        results.append({
            "Metric": name,
            "R^2": round(r2, 4),
            "Adjusted R^2": round(adj_r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Median AE": round(med_ae, 4),
            "Max Error": round(max_err, 4),
            "Mean Relative Error (%)": round(np.mean(rel_errors) * 100, 2)
        })

    # 출력
    summary_df = pd.DataFrame(results)
    print(summary_df)
    return summary_df

if __name__ == "__main__":
    path = os.sys.argv[1] if len(os.sys.argv) > 1 else "fittit_results.json"
    summarize_predictions_and_errors(path)
