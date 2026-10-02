from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

input_path = PROCESSED_DIR / "machinelearning_combined_clean.csv"

if not input_path.exists():
    raise FileNotFoundError(f"Missing input file: {input_path}")

print("Loading:", input_path)

df = pd.read_csv(input_path, low_memory=False)

print("Loaded shape:", df.shape)

if "Label" not in df.columns:
    raise ValueError("Expected column 'Label' was not found.")

# Fix label encoding artifacts from source file decoding.
label_fixes = {
    "Web Attack � Brute Force": "Web Attack - Brute Force",
    "Web Attack � XSS": "Web Attack - XSS",
    "Web Attack � Sql Injection": "Web Attack - Sql Injection",
}

df["Label"] = df["Label"].replace(label_fixes)

# Remove rows with missing labels, if any.
missing_labels = int(df["Label"].isna().sum())
df = df.dropna(subset=["Label"]).copy()

# Create binary label for IDS detection:
# 0 = BENIGN, 1 = ATTACK
df["binary_label"] = np.where(df["Label"] == "BENIGN", 0, 1)

# Keep multiclass label for reporting later.
df["attack_label"] = df["Label"]

# Drop non-model audit column from features.
non_feature_cols = ["Label", "binary_label", "attack_label"]
if "source_file" in df.columns:
    non_feature_cols.append("source_file")

feature_cols = [c for c in df.columns if c not in non_feature_cols]

# Convert features to numeric.
# Any non-numeric parsing failures become NaN.
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinity values with NaN.
inf_count = int(np.isinf(df[feature_cols]).sum().sum())
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

missing_feature_values = int(df[feature_cols].isna().sum().sum())

print("Feature columns:", len(feature_cols))
print("Infinity values replaced:", inf_count)
print("Missing feature values after conversion:", missing_feature_values)

# Build a reproducible modeling population.
# We keep all rare classes and sample large classes to make modeling practical.
RANDOM_SEED = 42
MAX_PER_CLASS = 60000
MIN_KEEP_ALL_THRESHOLD = 60000

sample_parts = []

for label, group in df.groupby("attack_label", sort=True):
    n = len(group)

    if n <= MIN_KEEP_ALL_THRESHOLD:
        take = n
        part = group
    else:
        take = MAX_PER_CLASS
        part = group.sample(n=take, random_state=RANDOM_SEED)

    print(f"Class: {label:35s} original={n:8d} selected={take:8d}")
    sample_parts.append(part)

sample_df = pd.concat(sample_parts, ignore_index=True)

# Shuffle after class-preserving sampling.
sample_df = sample_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

print("\nModeling sample shape:", sample_df.shape)

# Split 60 / 20 / 20.
# Stratify by attack_label so rare classes are preserved as much as possible.
train_df, temp_df = train_test_split(
    sample_df,
    test_size=0.40,
    random_state=RANDOM_SEED,
    stratify=sample_df["attack_label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["attack_label"]
)

print("\nSplit sizes:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

# Save split files.
sample_out = PROCESSED_DIR / "modeling_sample.csv"
train_out = PROCESSED_DIR / "train.csv"
val_out = PROCESSED_DIR / "validation.csv"
test_out = PROCESSED_DIR / "test.csv"

sample_df.to_csv(sample_out, index=False)
train_df.to_csv(train_out, index=False)
val_df.to_csv(val_out, index=False)
test_df.to_csv(test_out, index=False)

# Save feature list.
feature_list_out = PROCESSED_DIR / "feature_columns.json"
with feature_list_out.open("w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)

# Save distributions.
def distribution_table(frame, name):
    dist = frame["attack_label"].value_counts().reset_index()
    dist.columns = ["attack_label", "count"]
    dist["split"] = name
    return dist

dist_df = pd.concat([
    distribution_table(sample_df, "sample"),
    distribution_table(train_df, "train"),
    distribution_table(val_df, "validation"),
    distribution_table(test_df, "test"),
], ignore_index=True)

dist_out = METRICS_DIR / "phase0_split_class_distribution.csv"
dist_df.to_csv(dist_out, index=False)

summary = {
    "input_file": str(input_path),
    "random_seed": RANDOM_SEED,
    "max_per_class": MAX_PER_CLASS,
    "rows_after_missing_label_drop": int(len(df)),
    "missing_labels_removed": missing_labels,
    "feature_columns": int(len(feature_cols)),
    "infinity_values_replaced": inf_count,
    "missing_feature_values_after_conversion": missing_feature_values,
    "modeling_sample_rows": int(len(sample_df)),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "binary_label_definition": {
        "0": "BENIGN",
        "1": "any attack label"
    },
    "label_fixes_applied": label_fixes,
}

summary_out = METRICS_DIR / "phase0_modeling_split_summary.json"

with summary_out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved:")
print(" ", sample_out)
print(" ", train_out)
print(" ", val_out)
print(" ", test_out)
print(" ", feature_list_out)
print(" ", dist_out)
print(" ", summary_out)

print("\nPhase 0D modeling sample and split complete.")
