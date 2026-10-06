import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = r"C:\Users\Vivek Akhil\Downloads\ai 2020.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# 1. Feature Engineering
df["Temp_Difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Mechanical_Power"] = df["Torque [Nm]"] * (
    df["Rotational speed [rpm]"] * (2 * np.pi / 60)
)
# Synthetic operational RUL (cycles until wear out threshold ~240 min)
df["Simulated_RUL"] = np.maximum(
    0, 240 - df["Tool wear [min]"] + np.random.normal(0, 5, len(df))
)

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Difference",
    "Mechanical_Power",
]
categorical_features = ["Type"]

X = df[numeric_features + categorical_features]
y_failure = df["Machine failure"]
y_rul = df["Simulated_RUL"]

X_train, X_test, y_f_train, y_f_test, y_r_train, y_r_test = train_test_split(
    X, y_failure, y_rul, test_size=0.2, random_state=42, stratify=y_failure
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

# Train Classification Pipeline
clf_pipeline = Pipeline(
    [
        ("prep", preprocessor),
        (
            "clf",
            HistGradientBoostingClassifier(
                class_weight="balanced", random_state=42
            ),
        ),
    ]
)
clf_pipeline.fit(X_train, y_f_train)

# Train RUL Regressor Pipeline
rul_pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("reg", HistGradientBoostingRegressor(random_state=42)),
    ]
)
rul_pipeline.fit(X_train, y_r_train)

# Compute Feature Importance for Explainability
perm_importance = permutation_importance(
    clf_pipeline, X_test, y_f_test, n_repeats=5, random_state=42
)
feature_names = numeric_features + ["Type_H", "Type_L", "Type_M"]
importance_dict = {
    feat: float(imp)
    for feat, imp in zip(numeric_features, perm_importance.importances_mean)
}

# Export all artifacts together
artifacts = {
    "classifier": clf_pipeline,
    "rul_model": rul_pipeline,
    "importance": importance_dict,
    "numeric_features": numeric_features,
}

joblib.dump(artifacts, "pdm_framework_artifacts.joblib")
print("Training complete: pdm_framework_artifacts.joblib saved.")