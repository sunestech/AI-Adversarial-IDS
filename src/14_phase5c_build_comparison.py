from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

required = [
    METRICS_DIR / "baseline_metrics.json",
    METRICS_DIR / "phase3_whitebox_full_summary.csv",
    METRICS_DIR / "phase4a_transfer_full_summary.csv",
    METRICS_DIR / "phase4b_hopskipjump_full_summary.csv",
    METRICS_DIR / "phase5a_feature_squeezing_summary.json",
    METRICS_DIR / "phase5a_feature_squeezing_attack_summary.csv",
    METRICS_DIR / "phase5b_adversarial_training_summary.json",
    METRICS_DIR / "phase5b_adversarial_training_attack_summary.csv",
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

with (METRICS_DIR / "baseline_metrics.json").open("r", encoding="utf-8") as f:
    baseline = json.load(f)

phase3 = pd.read_csv(METRICS_DIR / "phase3_whitebox_full_summary.csv")
phase4a = pd.read_csv(METRICS_DIR / "phase4a_transfer_full_summary.csv")
phase4b = pd.read_csv(METRICS_DIR / "phase4b_hopskipjump_full_summary.csv")

with (METRICS_DIR / "phase5a_feature_squeezing_summary.json").open("r", encoding="utf-8") as f:
    fs_json = json.load(f)

fs_attacks = pd.read_csv(METRICS_DIR / "phase5a_feature_squeezing_attack_summary.csv")

with (METRICS_DIR / "phase5b_adversarial_training_summary.json").open("r", encoding="utf-8") as f:
    at_json = json.load(f)

at_attacks = pd.read_csv(METRICS_DIR / "phase5b_adversarial_training_attack_summary.csv")


def get_evasion(df, method, eps):
    rows = df[
        (df["attack_method"] == method) &
        (df["constraint_mode"] == "constrained") &
        (df["epsilon"].round(3) == round(eps, 3))
    ]
    if len(rows) == 0:
        return None
    return float(rows.iloc[0]["evasion_rate"])


def get_fs_evasion(method, eps):
    rows = fs_attacks[
        (fs_attacks["attack_method"] == method) &
        (fs_attacks["epsilon"].round(3) == round(eps, 3))
    ]
    if len(rows) == 0:
        return None
    return float(rows.iloc[0]["evasion_rate"])


def get_at_evasion(method, eps):
    rows = at_attacks[
        (at_attacks["attack_method"] == method) &
        (at_attacks["epsilon"].round(3) == round(eps, 3))
    ]
    if len(rows) == 0:
        return None
    return float(rows.iloc[0]["evasion_rate"])


def get_transfer(eps, col):
    rows = phase4a[phase4a["epsilon"].round(3) == round(eps, 3)]
    if len(rows) == 0:
        return None
    return float(rows.iloc[0][col])


def get_hsj(eps):
    rows = phase4b[phase4b["epsilon_projection"].round(3) == round(eps, 3)]
    if len(rows) == 0:
        return None
    return float(rows.iloc[0]["constrained_projected_evasion_rate"])


eps_report = 0.100

rows = []

rows.append({
    "scenario": "Baseline PyTorch IDS",
    "clean_accuracy": baseline["pytorch"]["accuracy"],
    "clean_attack_recall": baseline["pytorch"]["attack_recall"],
    "false_positive_rate": baseline["pytorch"]["false_positive_rate"],
    "fgsm_constrained_evasion_eps_0_100": get_evasion(phase3, "FGSM", eps_report),
    "pgd_constrained_evasion_eps_0_100": get_evasion(phase3, "PGD", eps_report),
    "transfer_pytorch_evasion_eps_0_100": get_transfer(eps_report, "pytorch_transfer_evasion_rate"),
    "transfer_xgboost_evasion_eps_0_100": get_transfer(eps_report, "xgboost_transfer_evasion_rate"),
    "hopskipjump_projected_evasion_eps_0_100": get_hsj(eps_report),
    "notes": "Frozen baseline model"
})

rows.append({
    "scenario": "Feature squeezing defense",
    "clean_accuracy": fs_json["clean_metrics_after_squeezing"]["accuracy"],
    "clean_attack_recall": fs_json["clean_metrics_after_squeezing"]["attack_recall"],
    "false_positive_rate": fs_json["clean_metrics_after_squeezing"]["false_positive_rate"],
    "fgsm_constrained_evasion_eps_0_100": get_fs_evasion("FGSM", eps_report),
    "pgd_constrained_evasion_eps_0_100": get_fs_evasion("PGD", eps_report),
    "transfer_pytorch_evasion_eps_0_100": None,
    "transfer_xgboost_evasion_eps_0_100": None,
    "hopskipjump_projected_evasion_eps_0_100": None,
    "notes": "Feature quantization applied before prediction"
})

rows.append({
    "scenario": "Adversarial training defense",
    "clean_accuracy": at_json["clean_metrics_after_defense"]["accuracy"],
    "clean_attack_recall": at_json["clean_metrics_after_defense"]["attack_recall"],
    "false_positive_rate": at_json["clean_metrics_after_defense"]["false_positive_rate"],
    "fgsm_constrained_evasion_eps_0_100": get_at_evasion("FGSM", eps_report),
    "pgd_constrained_evasion_eps_0_100": get_at_evasion("PGD", eps_report),
    "transfer_pytorch_evasion_eps_0_100": None,
    "transfer_xgboost_evasion_eps_0_100": None,
    "hopskipjump_projected_evasion_eps_0_100": None,
    "notes": "Retrained with clean plus constrained PGD adversarial examples"
})

comparison = pd.DataFrame(rows)

out_csv = METRICS_DIR / "final_defense_comparison_table.csv"
out_json = METRICS_DIR / "final_defense_comparison_table.json"
complete_out = METRICS_DIR / "phase5c_comparison_complete.txt"

comparison.to_csv(out_csv, index=False)

with out_json.open("w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

complete_out.write_text(
    "Phase 5C defense comparison table complete.\n",
    encoding="utf-8"
)

print("Final defense comparison table:")
print(comparison.to_string(index=False))

print("\nSaved:")
print(" ", out_csv)
print(" ", out_json)
print(" ", complete_out)

print("\nPhase 5C complete.")
