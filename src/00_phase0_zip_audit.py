from pathlib import Path
import zipfile
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

zip_names = [
    "MachineLearningCSV.zip",
    "GeneratedLabelledFlows.zip",
    "pcap-20260928T204621Z-1-001.zip",
]

audit_rows = []

for zip_name in zip_names:
    zip_path = RAW_DIR / zip_name

    if not zip_path.exists():
        print(f"MISSING: {zip_path}")
        continue

    print(f"\nInspecting: {zip_name}")

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        csv_members = [m for m in members if m.lower().endswith(".csv")]
        pcap_members = [
            m for m in members
            if m.lower().endswith(".pcap") or m.lower().endswith(".pcapng")
        ]
        md5_members = [m for m in members if m.lower().endswith(".md5")]

        print(f"  Total members: {len(members)}")
        print(f"  CSV files:     {len(csv_members)}")
        print(f"  PCAP files:    {len(pcap_members)}")
        print(f"  MD5 files:     {len(md5_members)}")

        if csv_members:
            for member in csv_members:
                with z.open(member) as f:
                    text = (line.decode("utf-8", errors="replace") for line in f)
                    reader = csv.reader(text)

                    try:
                        header = next(reader)
                    except StopIteration:
                        header = []

                    row_count = sum(1 for _ in reader)

                label_cols = [c for c in header if c.strip().lower() == "label"]

                print(f"  CSV: {member}")
                print(f"       rows without header: {row_count}")
                print(f"       columns:             {len(header)}")
                print(f"       label columns:        {label_cols}")

                audit_rows.append({
                    "zip_file": zip_name,
                    "member": member,
                    "member_type": "csv",
                    "rows_without_header": row_count,
                    "columns": len(header),
                    "label_columns": ";".join(label_cols),
                })

        for member in pcap_members:
            audit_rows.append({
                "zip_file": zip_name,
                "member": member,
                "member_type": "pcap",
                "rows_without_header": "",
                "columns": "",
                "label_columns": "",
            })

        for member in md5_members:
            audit_rows.append({
                "zip_file": zip_name,
                "member": member,
                "member_type": "md5",
                "rows_without_header": "",
                "columns": "",
                "label_columns": "",
            })

csv_out = METRICS_DIR / "phase0_zip_audit.csv"
json_out = METRICS_DIR / "phase0_zip_audit.json"

fieldnames = [
    "zip_file",
    "member",
    "member_type",
    "rows_without_header",
    "columns",
    "label_columns",
]

with csv_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(audit_rows)

with json_out.open("w", encoding="utf-8") as f:
    json.dump(audit_rows, f, indent=2)

print("\nSaved:")
print(f"  {csv_out}")
print(f"  {json_out}")
print("\nPhase 0 ZIP audit complete.")
