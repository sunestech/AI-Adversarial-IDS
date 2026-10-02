from pathlib import Path
import json
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

required = [
    METRICS_DIR / "baseline_metrics.json",
    METRICS_DIR / "phase3_whitebox_full_summary.csv",
    METRICS_DIR / "phase4a_transfer_full_summary.csv",
    METRICS_DIR / "phase4b_hopskipjump_full_summary.csv",
    METRICS_DIR / "final_defense_comparison_table.csv",
    METRICS_DIR / "feature_feasibility_table.csv",
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

with (METRICS_DIR / "baseline_metrics.json").open("r", encoding="utf-8") as f:
    baseline = json.load(f)

phase3 = pd.read_csv(METRICS_DIR / "phase3_whitebox_full_summary.csv")
phase4a = pd.read_csv(METRICS_DIR / "phase4a_transfer_full_summary.csv")
phase4b = pd.read_csv(METRICS_DIR / "phase4b_hopskipjump_full_summary.csv")
comparison = pd.read_csv(METRICS_DIR / "final_defense_comparison_table.csv")
feasibility = pd.read_csv(METRICS_DIR / "feature_feasibility_table.csv")

def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"

def phase3_value(method, mode, eps):
    row = phase3[
        (phase3["attack_method"] == method) &
        (phase3["constraint_mode"] == mode) &
        (phase3["epsilon"].round(3) == round(eps, 3))
    ]
    if len(row) == 0:
        return None
    return float(row.iloc[0]["evasion_rate"])

def transfer_value(eps, col):
    row = phase4a[phase4a["epsilon"].round(3) == round(eps, 3)]
    if len(row) == 0:
        return None
    return float(row.iloc[0][col])

def hsj_value(eps):
    row = phase4b[phase4b["epsilon_projection"].round(3) == round(eps, 3)]
    if len(row) == 0:
        return None
    return float(row.iloc[0]["constrained_projected_evasion_rate"])

eps = 0.100

fgsm_uncon = phase3_value("FGSM", "unconstrained", eps)
fgsm_con = phase3_value("FGSM", "constrained", eps)
pgd_uncon = phase3_value("PGD", "unconstrained", eps)
pgd_con = phase3_value("PGD", "constrained", eps)

pytorch_transfer = transfer_value(eps, "pytorch_transfer_evasion_rate")
xgb_transfer = transfer_value(eps, "xgboost_transfer_evasion_rate")
hsj_projected = hsj_value(eps)

perturbable = feasibility[feasibility["perturbation_allowed"] == 1]["feature"].tolist()

comparison_md = comparison.to_markdown(index=False)

report = f"""# Adversarial Machine Learning Assessment of an IDS Classifier

Generated: {datetime.now().isoformat(timespec="seconds")}

## 1. Executive Summary

This project built and evaluated a machine-learning intrusion detection system under adversarial evasion conditions. The baseline IDS performed strongly on clean data, but unconstrained white-box attacks were able to evade the PyTorch model at very high rates. When perturbations were restricted to a conservative set of operationally plausible network-flow features, evasion dropped substantially, showing that feasibility constraints materially change the security interpretation.

At epsilon {eps:.3f}, unconstrained PGD evasion reached {pct(pgd_uncon)}, while constrained PGD evasion was {pct(pgd_con)}. The black-box transfer attack produced {pct(pytorch_transfer)} PyTorch transfer evasion and {pct(xgb_transfer)} XGBoost transfer evasion at epsilon {eps:.3f}. HopSkipJump produced raw adversarial examples, but projected constrained evasion was {pct(hsj_projected)} at epsilon {eps:.3f}. Adversarial training was the strongest tested defense, reducing constrained FGSM and PGD evasion to the low single digits while preserving high attack recall.

## 2. Threat Model

The attacker goal is evasion: real attack traffic is modified at inference time so that the IDS predicts it as benign. The project evaluated three levels of attacker knowledge:

- White-box: the attacker has access to model architecture, weights, and gradients.
- Black-box transfer: the attacker trains a separate surrogate model and transfers adversarial examples to the protected model.
- Decision-based black-box: the attacker only observes predicted labels from the target model.

The attacker is not allowed to change arbitrary features. Network-flow features must remain within scaled bounds and, for constrained attacks, only attacker-controllable features may be perturbed.

## 3. System Under Test

The protected IDS system includes two clean baseline models:

- PyTorch neural-network IDS used as the differentiable white-box target.
- XGBoost IDS used as a stronger non-neural comparison model.

The PyTorch baseline metrics were:

- Clean accuracy: {pct(baseline["pytorch"]["accuracy"])}
- Attack precision: {pct(baseline["pytorch"]["attack_precision"])}
- Attack recall: {pct(baseline["pytorch"]["attack_recall"])}
- Attack F1: {pct(baseline["pytorch"]["attack_f1"])}
- False positive rate: {pct(baseline["pytorch"]["false_positive_rate"])}

The XGBoost baseline metrics were:

- Clean accuracy: {pct(baseline["xgboost"]["accuracy"])}
- Attack precision: {pct(baseline["xgboost"]["attack_precision"])}
- Attack recall: {pct(baseline["xgboost"]["attack_recall"])}
- Attack F1: {pct(baseline["xgboost"]["attack_f1"])}
- False positive rate: {pct(baseline["xgboost"]["false_positive_rate"])}

## 4. Dataset and Preprocessing

The modeling pipeline used the supplied MachineLearningCSV flow records. The processed dataset was split into training, validation, and test partitions. The fitted preprocessing objects were frozen after baseline training.

Preprocessing controls:

- Median imputation fitted on training data only.
- MinMax scaling fitted on training data only.
- Test set preserved as the downstream reference set.
- Scaled feature range clipped to [0, 1].
- Baseline artifacts frozen with SHA256 manifests.

## 5. Attack Surface and Feasibility Analysis

The feasibility analysis classified 78 model input features. Only 5 were allowed to change during constrained attacks:

{chr(10).join(f"- {x}" for x in perturbable)}

The remaining 73 features were frozen because they were derived, server-side, context-dependent, protocol-state dependent, or not included in the conservative attacker-controllable set.

This distinction is central to the assessment. Unconstrained attacks show mathematical vulnerability. Constrained attacks provide the more meaningful operational security result.

## 6. White-Box Attack Results

White-box attacks were run against the PyTorch IDS using FGSM and PGD across multiple epsilon values.

At epsilon {eps:.3f}:

- FGSM unconstrained evasion: {pct(fgsm_uncon)}
- FGSM constrained evasion: {pct(fgsm_con)}
- PGD unconstrained evasion: {pct(pgd_uncon)}
- PGD constrained evasion: {pct(pgd_con)}

The gap between unconstrained and constrained results shows that many mathematically successful adversarial perturbations depend on changing features outside the conservative attacker-controllable set.

## 7. Black-Box Attack Results

### Transfer Attack

The transfer attack trained an independent surrogate network, crafted constrained PGD examples against the surrogate, and submitted the same examples to the protected PyTorch and XGBoost models.

At epsilon {eps:.3f}:

- PyTorch transfer evasion: {pct(pytorch_transfer)}
- XGBoost transfer evasion: {pct(xgb_transfer)}

This confirms that some adversarial risk transfers across models even without direct access to protected-model weights.

### HopSkipJump Decision-Based Attack

HopSkipJump generated raw decision-based adversarial examples. After projecting the examples back into the constrained feature space, projected evasion at epsilon {eps:.3f} was:

- HopSkipJump projected evasion: {pct(hsj_projected)}

This result indicates that the raw decision-based attack found mathematical adversarial examples, but the conservative feasibility projection removed the evasion effect in the tested configuration.

## 8. Defense Results

Two defenses were tested:

- Feature squeezing
- Adversarial training

The main comparison table is:

{comparison_md}

Adversarial training was the strongest defense in this experiment. It reduced constrained FGSM and PGD evasion at epsilon {eps:.3f} while preserving high attack recall.

## 9. Residual Risk

The IDS remains vulnerable under some adversarial conditions. In particular, constrained white-box attacks and transfer attacks still produced nonzero evasion at larger perturbation budgets. A realistic adversary may also discover additional controllable feature pathways beyond the conservative five-feature threat model used here.

The largest residual concern is not the raw unconstrained evasion rate; it is the portion of evasion that survives feasibility constraints and transfers across model families.

## 10. Limitations

This assessment has several limitations:

- The feasibility model is conservative and may exclude some features a skilled attacker could influence.
- The dataset is static and may not reflect current network behavior.
- Flow-level features are abstractions of packet behavior.
- PCAP files were not available in the supplied PCAP archive; that archive contained checksum files only.
- HopSkipJump was evaluated on a bounded subset because query-based attacks are computationally expensive.
- Defenses were evaluated experimentally, not deployed in a real-time IDS pipeline.

## 11. Future Work

Recommended next steps:

- Extend the feasibility model with packet-level validation.
- Add real PCAP-derived replay or simulation testing.
- Evaluate additional black-box attacks such as ZOO.
- Test combined defenses: adversarial training plus feature squeezing.
- Build a Streamlit demo showing clean, adversarial, and defended predictions.
- Expand reporting by attack family, especially for rare classes.

## 12. Conclusion

The project demonstrates that adversarial evasion is a real risk for machine-learning IDS models, but the magnitude of the risk depends heavily on whether perturbations are operationally feasible. Unconstrained white-box attacks produced severe evasion. Constrained attacks were weaker but still meaningful at larger epsilon values. Black-box transfer attacks also produced measurable evasion. Defensive retraining substantially reduced constrained evasion and provides the strongest mitigation observed in this experiment.
"""

out = REPORT_DIR / "final_report.md"
out.write_text(report, encoding="utf-8")

complete = METRICS_DIR / "phase6_final_report_complete.txt"
complete.write_text("Phase 6 final report draft complete.\n", encoding="utf-8")

print("Final report draft saved:")
print(" ", out)
print(" ", complete)
print("\nPhase 6 final report draft complete.")
