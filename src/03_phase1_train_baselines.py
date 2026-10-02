from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    average_precision_score
)

from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

train_path = PROCESSED_DIR / "train.csv"
val_path = PROCESSED_DIR / "validation.csv"
test_path = PROCESSED_DIR / "test.csv"
feature_path = PROCESSED_DIR / "feature_columns.json"

for path in [train_path, val_path, test_path, feature_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Load data
# -----------------------------
print("Loading split files...")

train_df = pd.read_csv(train_path, low_memory=False)
val_df = pd.read_csv(val_path, low_memory=False)
test_df = pd.read_csv(test_path, low_memory=False)

with feature_path.open("r", encoding="utf-8") as f:
    feature_cols = json.load(f)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)
print("Feature columns:", len(feature_cols))

target_col = "binary_label"

for df_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    if target_col not in df.columns:
        raise ValueError(f"{df_name} is missing {target_col}")


# -----------------------------
# Build X and y
# -----------------------------
X_train_raw = train_df[feature_cols]
X_val_raw = val_df[feature_cols]
X_test_raw = test_df[feature_cols]

y_train = train_df[target_col].astype(int).to_numpy()
y_val = val_df[target_col].astype(int).to_numpy()
y_test = test_df[target_col].astype(int).to_numpy()

attack_label_test = test_df["attack_label"].astype(str).to_numpy()

print("\nBinary class counts:")
print("Train:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Validation:", dict(zip(*np.unique(y_val, return_counts=True))))
print("Test:", dict(zip(*np.unique(y_test, return_counts=True))))


# -----------------------------
# Fit preprocessing on training only
# -----------------------------
print("\nFitting imputer and MinMaxScaler on training data only...")

imputer = SimpleImputer(strategy="median")
scaler = MinMaxScaler()

X_train_imp = imputer.fit_transform(X_train_raw)
X_val_imp = imputer.transform(X_val_raw)
X_test_imp = imputer.transform(X_test_raw)

X_train = scaler.fit_transform(X_train_imp)
X_val = scaler.transform(X_val_imp)
X_test = scaler.transform(X_test_imp)

# Clip small numeric spillover outside [0,1]
X_train = np.clip(X_train, 0.0, 1.0).astype(np.float32)
X_val = np.clip(X_val, 0.0, 1.0).astype(np.float32)
X_test = np.clip(X_test, 0.0, 1.0).astype(np.float32)

print("Scaled train min/max:", float(X_train.min()), float(X_train.max()))
print("Scaled val min/max:", float(X_val.min()), float(X_val.max()))
print("Scaled test min/max:", float(X_test.min()), float(X_test.max()))

joblib.dump(imputer, MODEL_DIR / "median_imputer.joblib")
joblib.dump(scaler, MODEL_DIR / "minmax_scaler.joblib")

with (MODEL_DIR / "feature_columns.json").open("w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)


# -----------------------------
# Save arrays for later phases
# -----------------------------
arrays_dir = PROCESSED_DIR / "baseline_arrays"
arrays_dir.mkdir(parents=True, exist_ok=True)

np.save(arrays_dir / "X_train_scaled.npy", X_train)
np.save(arrays_dir / "X_val_scaled.npy", X_val)
np.save(arrays_dir / "X_test_scaled.npy", X_test)
np.save(arrays_dir / "y_train.npy", y_train)
np.save(arrays_dir / "y_val.npy", y_val)
np.save(arrays_dir / "y_test.npy", y_test)

pd.DataFrame({"attack_label": attack_label_test}).to_csv(
    arrays_dir / "test_attack_labels.csv",
    index=False
)


# -----------------------------
# Metrics helper
# -----------------------------
def binary_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0
    )

    # F2 for attack class only
    attack_precision = precision[1]
    attack_recall = recall[1]
    if attack_precision + attack_recall == 0:
        f2_attack = 0.0
    else:
        f2_attack = (5 * attack_precision * attack_recall) / ((4 * attack_precision) + attack_recall)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    result = {
        "accuracy": float(acc),
        "benign_precision": float(precision[0]),
        "benign_recall": float(recall[0]),
        "benign_f1": float(f1[0]),
        "benign_support": int(support[0]),
        "attack_precision": float(precision[1]),
        "attack_recall": float(recall[1]),
        "attack_f1": float(f1[1]),
        "attack_f2": float(f2_attack),
        "attack_support": int(support[1]),
        "false_positive_rate": float(fpr),
        "confusion_matrix_labels": ["BENIGN", "ATTACK"],
        "confusion_matrix": cm.tolist(),
    }

    if y_score is not None:
        result["attack_pr_auc"] = float(average_precision_score(y_true, y_score))

    return result


def save_classification_report(path, y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
    path.write_text(report, encoding="utf-8")


# -----------------------------
# PyTorch model
# -----------------------------
class IDSClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train.shape[1]
model = IDSClassifier(input_dim=input_dim).to(device)

# Class weights for imbalance
class_counts = np.bincount(y_train, minlength=2)
class_weights = class_counts.sum() / (2.0 * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

val_x_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
val_y_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)

print("\nTraining PyTorch baseline IDS...")

best_val_f1 = -1.0
best_state = None
history = []

EPOCHS = 20

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_ds)

    model.eval()
    with torch.no_grad():
        val_logits = model(val_x_tensor)
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()

    val_metrics = binary_metrics(y_val, val_pred, val_probs)
    val_attack_f1 = val_metrics["attack_f1"]

    history.append({
        "epoch": epoch,
        "train_loss": float(avg_loss),
        "val_attack_precision": val_metrics["attack_precision"],
        "val_attack_recall": val_metrics["attack_recall"],
        "val_attack_f1": val_metrics["attack_f1"],
        "val_accuracy": val_metrics["accuracy"],
    })

    print(
        f"Epoch {epoch:02d} | loss={avg_loss:.6f} | "
        f"val_acc={val_metrics['accuracy']:.6f} | "
        f"val_attack_recall={val_metrics['attack_recall']:.6f} | "
        f"val_attack_f1={val_metrics['attack_f1']:.6f}"
    )

    if val_attack_f1 > best_val_f1:
        best_val_f1 = val_attack_f1
        best_state = {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "feature_columns": feature_cols,
            "class_weights": class_weights.tolist(),
            "random_seed": RANDOM_SEED,
        }

torch.save(best_state, MODEL_DIR / "pytorch_clean_baseline.pt")

# Load best model for test evaluation
model.load_state_dict(best_state["model_state_dict"])
model.eval()

test_x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    test_logits = model(test_x_tensor)
    test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
    test_pred_torch = torch.argmax(test_logits, dim=1).cpu().numpy()

torch_metrics = binary_metrics(y_test, test_pred_torch, test_probs)
save_classification_report(METRICS_DIR / "pytorch_classification_report.txt", y_test, test_pred_torch)

pd.DataFrame(history).to_csv(METRICS_DIR / "pytorch_training_history.csv", index=False)
np.save(arrays_dir / "pytorch_test_predictions.npy", test_pred_torch)
np.save(arrays_dir / "pytorch_test_attack_scores.npy", test_probs)


# -----------------------------
# XGBoost model
# -----------------------------
print("\nTraining XGBoost baseline IDS...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_probs >= 0.5).astype(int)

xgb_metrics = binary_metrics(y_test, xgb_pred, xgb_probs)
save_classification_report(METRICS_DIR / "xgboost_classification_report.txt", y_test, xgb_pred)

xgb_model.save_model(MODEL_DIR / "xgboost_clean_baseline.json")
np.save(arrays_dir / "xgboost_test_predictions.npy", xgb_pred)
np.save(arrays_dir / "xgboost_test_attack_scores.npy", xgb_probs)


# -----------------------------
# Originally correct attack cohorts
# -----------------------------
attack_mask = y_test == 1

pytorch_correct_attack_indices = np.where(attack_mask & (test_pred_torch == 1))[0]
xgboost_correct_attack_indices = np.where(attack_mask & (xgb_pred == 1))[0]
shared_correct_attack_indices = np.where(
    attack_mask & (test_pred_torch == 1) & (xgb_pred == 1)
)[0]

np.save(arrays_dir / "pytorch_correct_attack_test_indices.npy", pytorch_correct_attack_indices)
np.save(arrays_dir / "xgboost_correct_attack_test_indices.npy", xgboost_correct_attack_indices)
np.save(arrays_dir / "shared_correct_attack_test_indices.npy", shared_correct_attack_indices)

print("\nOriginally correct attack cohorts:")
print("PyTorch correct attack rows:", len(pytorch_correct_attack_indices))
print("XGBoost correct attack rows:", len(xgboost_correct_attack_indices))
print("Shared correct attack rows:", len(shared_correct_attack_indices))


# -----------------------------
# Save metrics
# -----------------------------
baseline_metrics = {
    "random_seed": RANDOM_SEED,
    "input_dim": int(input_dim),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "feature_columns": int(len(feature_cols)),
    "preprocessing": {
        "imputer": "SimpleImputer(strategy='median') fitted on train only",
        "scaler": "MinMaxScaler() fitted on train only",
        "clip_values": [0.0, 1.0]
    },
    "pytorch": torch_metrics,
    "xgboost": xgb_metrics,
    "originally_correct_attack_cohorts": {
        "pytorch_correct_attack_rows": int(len(pytorch_correct_attack_indices)),
        "xgboost_correct_attack_rows": int(len(xgboost_correct_attack_indices)),
        "shared_correct_attack_rows": int(len(shared_correct_attack_indices)),
    }
}

metrics_out = METRICS_DIR / "baseline_metrics.json"

with metrics_out.open("w", encoding="utf-8") as f:
    json.dump(baseline_metrics, f, indent=2)

# Completion marker
(METRICS_DIR / "phase1_baseline_complete.txt").write_text(
    "Phase 1 baseline training complete.\n",
    encoding="utf-8"
)

print("\nSaved baseline artifacts:")
print(" ", MODEL_DIR / "pytorch_clean_baseline.pt")
print(" ", MODEL_DIR / "xgboost_clean_baseline.json")
print(" ", MODEL_DIR / "median_imputer.joblib")
print(" ", MODEL_DIR / "minmax_scaler.joblib")
print(" ", METRICS_DIR / "baseline_metrics.json")
print(" ", METRICS_DIR / "phase1_baseline_complete.txt")

print("\nPyTorch test metrics:")
print(json.dumps(torch_metrics, indent=2))

print("\nXGBoost test metrics:")
print(json.dumps(xgb_metrics, indent=2))

print("\nPhase 1 baseline training complete.")
