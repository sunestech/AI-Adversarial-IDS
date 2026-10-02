from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)

feature_path = PROCESSED_DIR / "feature_columns.json"

if not feature_path.exists():
    raise FileNotFoundError(f"Missing feature list: {feature_path}")

with feature_path.open("r", encoding="utf-8") as f:
    feature_cols = json.load(f)

print("Loaded feature count:", len(feature_cols))

# Conservative attacker model:
# 1 = attacker can directly perturb this feature
# 0 = frozen during constrained attacks
directly_perturbable = {
    "Flow Duration": {
        "direction": "increase_or_small_adjustment",
        "constraint": "must remain >= 0 and within scaled epsilon budget",
        "reason": "attacker may delay or pace traffic, but cannot create impossible negative duration"
    },
    "Total Fwd Packets": {
        "direction": "increase",
        "constraint": "must remain >= 0 and integer-like after inverse transform",
        "reason": "attacker may add forward packets or split payloads, but cannot arbitrarily reduce already observed packets"
    },
    "Total Length of Fwd Packets": {
        "direction": "increase",
        "constraint": "must remain >= 0 and consistent with forward packet count",
        "reason": "attacker may pad or alter forward payload size within realistic limits"
    },
    "Init_Win_bytes_forward": {
        "direction": "bounded_adjustment",
        "constraint": "must remain >= 0 and within valid TCP window-size range",
        "reason": "forward TCP initial window can vary by host stack and attacker-controlled client behavior"
    },
    "act_data_pkt_fwd": {
        "direction": "increase_or_same",
        "constraint": "must remain >= 0 and integer-like after inverse transform",
        "reason": "attacker may influence number of forward packets carrying data"
    },
}

# Heuristic categories for the remaining flow features.
def classify_feature(feature):
    if feature in directly_perturbable:
        meta = directly_perturbable[feature]
        return {
            "attacker_controllable": "yes",
            "category": "directly_perturbable",
            "perturbation_allowed": 1,
            "direction": meta["direction"],
            "constraint": meta["constraint"],
            "reason": meta["reason"],
        }

    name = feature.lower()

    if "label" in name:
        category = "target_or_label"
        reason = "label is not an input feature and must never be perturbed"
    elif "id" == name or name.endswith("_id"):
        category = "identity_or_index"
        reason = "identifier or index fields are not behavioral network-flow measurements"
    elif "bwd" in name or "backward" in name:
        category = "context_or_server_side"
        reason = "backward/server-to-client behavior is not directly controlled by the attacker in this conservative model"
    elif "min" in name or "max" in name or "mean" in name or "std" in name or "avg" in name or "variance" in name:
        category = "derived_or_statistical"
        reason = "statistical aggregate depends on multiple packet-level events and should not be independently perturbed"
    elif "iat" in name:
        category = "derived_or_timing_aggregate"
        reason = "inter-arrival timing is derived from packet timestamps and must remain internally consistent"
    elif "bulk" in name or "subflow" in name:
        category = "derived_or_flow_aggregate"
        reason = "bulk and subflow values are derived aggregates and should not be independently changed"
    elif "flag" in name or "urg" in name or "psh" in name or "ece" in name or "cwe" in name or "rst" in name or "ack" in name or "syn" in name or "fin" in name:
        category = "protocol_state"
        reason = "TCP flag counts must correspond to a valid packet sequence, so they are frozen in this conservative model"
    elif "header" in name:
        category = "protocol_or_derived_size"
        reason = "header length is protocol-dependent and must remain consistent with packet structure"
    elif "packet length" in name or "pkt len" in name or "fwd packet length" in name:
        category = "derived_or_dependent"
        reason = "packet length summary fields depend on packet population and should not be changed independently"
    elif "flow bytes" in name or "flow packets" in name:
        category = "derived_rate"
        reason = "flow rates are derived from duration and packet or byte counts"
    else:
        category = "frozen"
        reason = "not included in the conservative attacker-controllable set"

    return {
        "attacker_controllable": "no",
        "category": category,
        "perturbation_allowed": 0,
        "direction": "none",
        "constraint": "feature frozen during constrained attacks",
        "reason": reason,
    }

rows = []

for idx, feature in enumerate(feature_cols):
    info = classify_feature(feature)
    rows.append({
        "feature_index": idx,
        "feature": feature,
        "attacker_controllable": info["attacker_controllable"],
        "category": info["category"],
        "perturbation_allowed": info["perturbation_allowed"],
        "direction": info["direction"],
        "constraint": info["constraint"],
        "reason": info["reason"],
    })

table = pd.DataFrame(rows)

out_csv = METRICS_DIR / "feature_feasibility_table.csv"
table.to_csv(out_csv, index=False)

summary = {
    "feature_count": int(len(feature_cols)),
    "directly_perturbable_count": int(table["perturbation_allowed"].sum()),
    "frozen_count": int((table["perturbation_allowed"] == 0).sum()),
    "directly_perturbable_features": table.loc[
        table["perturbation_allowed"] == 1,
        "feature"
    ].tolist(),
    "attacker_model": "conservative network-flow evasion model",
    "purpose": "constrain adversarial examples so perturbations affect only plausible attacker-controllable flow features",
}

summary_out = METRICS_DIR / "phase2_feasibility_summary.json"

with summary_out.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

complete_out = METRICS_DIR / "phase2_feasibility_complete.txt"
complete_out.write_text(
    "Phase 2 feature feasibility table complete.\n",
    encoding="utf-8"
)

print("Phase 2 feasibility table complete.")
print("Feature count:", summary["feature_count"])
print("Directly perturbable:", summary["directly_perturbable_count"])
print("Frozen:", summary["frozen_count"])
print("\nDirectly perturbable features:")
for feature in summary["directly_perturbable_features"]:
    print(" -", feature)

print("\nSaved:")
print(" ", out_csv)
print(" ", summary_out)
print(" ", complete_out)
