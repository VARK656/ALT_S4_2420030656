import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Dynamic Path Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'ai 2020.csv'))
MODEL_PATH = os.path.join(BASE_DIR, 'pdm_framework_artifacts.joblib')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH}\nEnsure 'ai 2020.csv' is inside the 'data' folder.")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# --- Feature Engineering ---
df["Temp_Difference"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Mechanical_Power"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * (2 * np.pi / 60))
df["Simulated_RUL"] = np.maximum(0, 240 - df["Tool wear [min]"] + np.random.normal(0, 5, len(df)))

numeric_features = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]", "Temp_Difference", "Mechanical_Power"
]
categorical_features = ["Type"]

X = df[numeric_features + categorical_features]
y_failure = df["Machine failure"]
y_rul = df["Simulated_RUL"]

X_train, X_test, y_f_train, y_f_test, y_r_train, y_r_test = train_test_split(
    X, y_failure, y_rul, test_size=0.2, random_state=42, stratify=y_failure
)

# --- Preprocessing & Pipelines ---
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
])

clf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=42)),
])
clf_pipeline.fit(X_train, y_f_train)

rul_pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg", HistGradientBoostingRegressor(random_state=42)),
])
rul_pipeline.fit(X_train, y_r_train)

# --- Explainability (Feature Importance) ---
print("Calculating feature importance...")
perm_importance = permutation_importance(clf_pipeline, X_test, y_f_test, n_repeats=5, random_state=42)
importance_dict = {feat: float(imp) for feat, imp in zip(numeric_features, perm_importance.importances_mean)}

# --- Export Artifacts ---
artifacts = {
    "classifier": clf_pipeline,
    "rul_model": rul_pipeline,
    "importance": importance_dict
}

joblib.dump(artifacts, MODEL_PATH)
print(f"Training complete. Model saved to: {MODEL_PATH}")