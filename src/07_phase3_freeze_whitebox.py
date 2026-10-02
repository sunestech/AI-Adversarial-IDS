from pathlib import Path
import hashlib
import json
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MANIFEST_DIR = PROJECT_ROOT / "results" / "manifests"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

required_files = [
    "results/metrics/feature_feasibility_table.csv",
    "results/metrics/phase2_feasibility_summary.json",
    "results/metrics/phase2_feasibility_complete.txt",
    "results/metrics/phase3_whitebox_full_summary.csv",
    "results/metrics/phase3_whitebox_full_summary.json",
    "results/metrics/phase3_whitebox_full_complete.txt",
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
    print("MISSING REQUIRED PHASE 3 FILES:")
    for item in missing:
        print(" -", item)
    raise SystemExit("Phase 3 freeze failed because required files are missing.")

manifest_df = pd.DataFrame(rows).sort_values("relative_path")

csv_out = MANIFEST_DIR / "phase3_whitebox_manifest.csv"
json_out = MANIFEST_DIR / "phase3_whitebox_manifest.json"

manifest_df.to_csv(csv_out, index=False)

with json_out.open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

complete_out = METRICS_DIR / "phase3_whitebox_frozen.txt"
complete_out.write_text(
    "Phase 3 white-box attack results frozen with SHA256 manifest.\n",
    encoding="utf-8"
)

print("Phase 3 freeze complete.")
print("Files hashed:", len(rows))
print("Saved:")
print(" ", csv_out)
print(" ", json_out)
print(" ", complete_out)
