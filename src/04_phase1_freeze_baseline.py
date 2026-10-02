from pathlib import Path
import hashlib
import json
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MANIFEST_DIR = PROJECT_ROOT / "results" / "manifests"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    "models/baseline/pytorch_clean_baseline.pt",
    "models/baseline/xgboost_clean_baseline.json",
    "models/baseline/median_imputer.joblib",
    "models/baseline/minmax_scaler.joblib",
    "models/baseline/feature_columns.json",

    "data/processed/baseline_arrays/X_train_scaled.npy",
    "data/processed/baseline_arrays/X_val_scaled.npy",
    "data/processed/baseline_arrays/X_test_scaled.npy",
    "data/processed/baseline_arrays/y_train.npy",
    "data/processed/baseline_arrays/y_val.npy",
    "data/processed/baseline_arrays/y_test.npy",

    "data/processed/baseline_arrays/pytorch_test_predictions.npy",
    "data/processed/baseline_arrays/pytorch_test_attack_scores.npy",
    "data/processed/baseline_arrays/xgboost_test_predictions.npy",
    "data/processed/baseline_arrays/xgboost_test_attack_scores.npy",

    "data/processed/baseline_arrays/pytorch_correct_attack_test_indices.npy",
    "data/processed/baseline_arrays/xgboost_correct_attack_test_indices.npy",
    "data/processed/baseline_arrays/shared_correct_attack_test_indices.npy",
    "data/processed/baseline_arrays/test_attack_labels.csv",

    "results/metrics/baseline_metrics.json",
    "results/metrics/pytorch_classification_report.txt",
    "results/metrics/xgboost_classification_report.txt",
    "results/metrics/pytorch_training_history.csv",
    "results/metrics/phase1_baseline_complete.txt",
]

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = []
missing = []

for rel_path in required_files:
    path = PROJECT_ROOT / rel_path

    if not path.exists():
        missing.append(rel_path)
        continue

    stat = path.stat()

    rows.append({
        "relative_path": rel_path,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc
        ).isoformat(),
        "sha256": sha256_file(path),
    })

if missing:
    print("MISSING REQUIRED BASELINE FILES:")
    for item in missing:
        print(" -", item)
    raise SystemExit("Baseline freeze failed because required files are missing.")

manifest_df = pd.DataFrame(rows).sort_values("relative_path")

csv_out = MANIFEST_DIR / "baseline_freeze_manifest.csv"
json_out = MANIFEST_DIR / "baseline_freeze_manifest.json"

manifest_df.to_csv(csv_out, index=False)

with json_out.open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

completion = METRICS_DIR / "phase1_baseline_frozen.txt"
completion.write_text(
    "Phase 1 baseline frozen with SHA256 manifest.\n",
    encoding="utf-8"
)

print("Baseline freeze complete.")
print("Files hashed:", len(rows))
print("Saved:")
print(" ", csv_out)
print(" ", json_out)
print(" ", completion)
