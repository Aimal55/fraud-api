import os
import ast
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import onnxmltools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
from onnxmltools.convert.common.data_types import FloatTensorType


# path
# Get the directory of the current 
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get parent directory 
parent_dir = os.path.dirname(script_dir)

# Path to transactions.csv
transactions_dir = os.path.join(parent_dir, "dataset", "transactions")
csv_path = os.path.join(transactions_dir, "transactions.csv")

os.makedirs(transactions_dir, exist_ok=True)

# Validate CSV presence
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"[ERROR] transactions.csv not found at: {csv_path}")
print(f"[INFO] CSV path resolved to: {csv_path}")

# Create saved_model directory to save models
savedmodel_dir = os.path.join(script_dir, "saved_model")
os.makedirs(savedmodel_dir, exist_ok=True)

# Load and preprocess

df = pd.read_csv(csv_path)
print(f"[INFO] Dataset loaded with shape: {df.shape}")

# Parse geo column back into lat/lon
geo_tuples = df["geo"].apply(ast.literal_eval)
df["geo_lat"] = geo_tuples.apply(lambda x: x[0])
df["geo_lon"] = geo_tuples.apply(lambda x: x[1])

# Device usage frequency Feature
device_freq = df["device_id"].value_counts()
df["device_tx_count"] = df["device_id"].map(device_freq)

# Geo distance  Feature
R = 6371  # Earth radius in km
lat1 = np.radians(25.0)
lon1 = np.radians(67.0)
lat2 = np.radians(df["geo_lat"])
lon2 = np.radians(df["geo_lon"])
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
df["geo_distance"] = R * c

# Feature Prep

X = df[["amount", "device_tx_count", "geo_distance"]].copy()
y = df["label"]

# Feature weighting
X["amount"] *= 2
X["geo_distance"] *= 1.5

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.30, random_state=42
)
print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler if needed later
joblib.dump(scaler, os.path.join(savedmodel_dir, "scaler.pkl"))

# XGBoost training

neg, pos = np.bincount(y_train.astype(int))
scale_pos_weight = neg / max(pos, 1)
print(f"[INFO] scale_pos_weight: {scale_pos_weight:.2f}")

clf = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_scaled, y_train)
print("[INFO] XGBoost model trained.")

# Evaluation

y_probs = clf.predict_proba(X_test_scaled)[:, 1]
pr_auc = average_precision_score(y_test, y_probs)
print(f"[INFO] PR-AUC on test set: {pr_auc:.4f}")

# Save model pkl

pkl_path = os.path.join(savedmodel_dir, "model.pkl")
joblib.dump(clf, pkl_path)
print(f"[INFO] Model saved to: {pkl_path}")

# Export to ONNX

initial_type = [('float_input', FloatTensorType([None, X_train_scaled.shape[1]]))]
onnx_model = onnxmltools.convert_xgboost(clf.get_booster(), initial_types=initial_type)

onnx_path = os.path.join(savedmodel_dir, "model.onnx")
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"[INFO] ONNX model saved to: {onnx_path}")
