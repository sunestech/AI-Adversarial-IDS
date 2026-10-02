from pathlib import Path
import zipfile
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

zip_path = RAW_DIR / "MachineLearningCSV.zip"

if not zip_path.exists():
    raise FileNotFoundError(f"Missing required file: {zip_path}")

print("Reading:", zip_path)

frames = []
file_summaries = []

with zipfile.ZipFile(zip_path, "r") as z:
    csv_members = sorted([
        m for m in z.namelist()
        if m.lower().endswith(".csv")
    ])

    print("CSV files found:", len(csv_members))

    for member in csv_members:
        print("\nLoading:", member)

        with z.open(member) as f:
            df_part = pd.read_csv(f, low_memory=False)

        original_rows = len(df_part)
        original_cols = len(df_part.columns)

        # Clean column names by removing leading/trailing spaces
        df_part.columns = [str(c).strip() for c in df_part.columns]

        # Add source file column for audit traceability
        df_part["source_file"] = member

        frames.append(df_part)

        label_col = "Label" if "Label" in df_part.columns else None

        label_counts = {}
        if label_col:
            label_counts = df_part[label_col].value_counts(dropna=False).to_dict()

        file_summaries.append({
            "source_file": member,
            "rows": int(original_rows),
            "columns_before_source_file": int(original_cols),
            "columns_after_source_file": int(len(df_part.columns)),
            "label_column_found": bool(label_col),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        })

combined = pd.concat(frames, ignore_index=True)

print("\nCombined shape before cleaning:", combined.shape)

# Replace infinite values with NaN
numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
inf_count = int(np.isinf(combined[numeric_cols]).sum().sum()) if numeric_cols else 0
combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], np.nan)

# Drop rows that are completely empty except source_file
cols_except_source = [c for c in combined.columns if c != "source_file"]
fully_empty_rows = int(combined[cols_except_source].isna().all(axis=1).sum())
combined = combined.loc[~combined[cols_except_source].isna().all(axis=1)].copy()

missing_numeric_after_inf = int(combined[numeric_cols].isna().sum().sum()) if numeric_cols else 0

if "Label" not in combined.columns:
    raise ValueError("Expected label column 'Label' was not found after column-name cleaning.")

class_distribution = combined["Label"].value_counts(dropna=False).reset_index()
class_distribution.columns = ["Label", "count"]

quality_report = {
    "input_zip": str(zip_path),
    "csv_files_loaded": len(csv_members),
    "combined_rows_after_cleaning": int(len(combined)),
    "combined_columns_after_cleaning": int(len(combined.columns)),
    "numeric_columns": int(len(numeric_cols)),
    "infinity_values_replaced_with_nan": inf_count,
    "fully_empty_rows_removed": fully_empty_rows,
    "missing_numeric_values_after_inf_replacement": missing_numeric_after_inf,
    "label_column": "Label",
    "unique_labels": int(combined["Label"].nunique(dropna=False)),
    "source_files": file_summaries,
}

combined_out = PROCESSED_DIR / "machinelearning_combined_clean.csv"
class_out = METRICS_DIR / "phase0_class_distribution.csv"
quality_json_out = METRICS_DIR / "phase0_data_quality_report.json"
quality_csv_out = METRICS_DIR / "phase0_file_summary.csv"

combined.to_csv(combined_out, index=False)
class_distribution.to_csv(class_out, index=False)
pd.DataFrame(file_summaries).drop(columns=["label_counts"]).to_csv(quality_csv_out, index=False)

with quality_json_out.open("w", encoding="utf-8") as f:
    json.dump(quality_report, f, indent=2)

print("\nSaved:")
print(" ", combined_out)
print(" ", class_out)
print(" ", quality_json_out)
print(" ", quality_csv_out)

print("\nFinal combined shape:", combined.shape)
print("\nClass distribution:")
print(class_distribution)

print("\nPhase 0C combine and clean complete.")
