from pathlib import Path
import re

path = Path("src/18_streamlit_dashboard.py")

if not path.exists():
    raise FileNotFoundError("Dashboard file not found: src/18_streamlit_dashboard.py")

text = path.read_text(encoding="utf-8")

# Remove Final Report from sidebar menu list.
text = text.replace(
'''        "Upload Metrics Explorer",
        "Final Report"''',
'''        "Upload Metrics Explorer"'''
)

# Replace the About Project page content with a shorter one-paragraph summary.
about_pattern = r'''# ------------------------------------------------------------
# About Project
# ------------------------------------------------------------
elif page == "About Project":
.*?
(?=# ------------------------------------------------------------
# Interactive Prediction Lab
# ------------------------------------------------------------)'''

about_replacement = '''# ------------------------------------------------------------
# About Project
# ------------------------------------------------------------
elif page == "About Project":
    st.header("About This Project")

    st.markdown("""
    **Author:** Sunday Esumike, PhD

    This project is an adversarial machine learning assessment of an Intrusion Detection System that builds clean PyTorch and XGBoost IDS baselines, evaluates white-box and black-box evasion attacks, applies realistic network-flow feasibility constraints, tests feature squeezing and adversarial training defenses, and summarizes the results in a security-research style workflow. The main finding is that unconstrained adversarial attacks can produce severe evasion, but feasibility constraints and adversarial training substantially reduce the operationally meaningful risk.
    """)

    st.subheader("Project Workflow")

    workflow = pd.DataFrame([
        {"Phase": "Phase 0", "Name": "Environment and Data Audit", "Purpose": "Verify resources, inspect ZIP files, combine and clean IDS flow data."},
        {"Phase": "Phase 1", "Name": "Baseline IDS Models", "Purpose": "Train PyTorch and XGBoost IDS baselines and freeze artifacts."},
        {"Phase": "Phase 2", "Name": "Feature Feasibility", "Purpose": "Decide which network-flow features an attacker can realistically perturb."},
        {"Phase": "Phase 3", "Name": "White-Box Attacks", "Purpose": "Run FGSM and PGD attacks in unconstrained and constrained modes."},
        {"Phase": "Phase 4A", "Name": "Transfer Attack", "Purpose": "Train a surrogate model and test black-box transfer evasion."},
        {"Phase": "Phase 4B", "Name": "HopSkipJump", "Purpose": "Run decision-based black-box attacks on a bounded subset."},
        {"Phase": "Phase 5", "Name": "Defenses", "Purpose": "Evaluate feature squeezing and adversarial training defenses."},
        {"Phase": "Dashboard", "Name": "Interactive Demo", "Purpose": "Load data, run IDS predictions, and explore attack and defense results interactively."},
    ])

    st.dataframe(workflow, use_container_width=True)

    st.subheader("Key Deliverables")

    deliverables = pd.DataFrame([
        {"Deliverable": "Defense comparison table", "Path": "results/metrics/final_defense_comparison_table.csv"},
        {"Deliverable": "Baseline metrics", "Path": "results/metrics/baseline_metrics.json"},
        {"Deliverable": "Feature feasibility table", "Path": "results/metrics/feature_feasibility_table.csv"},
        {"Deliverable": "White-box attack summary", "Path": "results/metrics/phase3_whitebox_full_summary.csv"},
        {"Deliverable": "Transfer attack summary", "Path": "results/metrics/phase4a_transfer_full_summary.csv"},
        {"Deliverable": "HopSkipJump summary", "Path": "results/metrics/phase4b_hopskipjump_full_summary.csv"},
        {"Deliverable": "PyTorch baseline model", "Path": "models/baseline/pytorch_clean_baseline.pt"},
        {"Deliverable": "XGBoost baseline model", "Path": "models/baseline/xgboost_clean_baseline.json"},
        {"Deliverable": "Adversarial training defense model", "Path": "models/defense/pytorch_adversarial_training_defense.pt"},
    ])

    st.dataframe(deliverables, use_container_width=True)


'''

text, count = re.subn(about_pattern, about_replacement, text, flags=re.S)

if count != 1:
    raise RuntimeError(f"Expected to replace 1 About Project block, replaced {count}.")

# Remove the Final Report page block completely.
final_report_pattern = r'''# ------------------------------------------------------------
# Final Report
# ------------------------------------------------------------
elif page == "Final Report":
.*?\Z'''

text, count = re.subn(final_report_pattern, "", text, flags=re.S)

# It is okay if count is 0 if the block was already removed.
path.write_text(text, encoding="utf-8")

print("Dashboard updated:")
print("- Removed Final Report page from sidebar")
print("- Shortened About Project to one paragraph")
print("- Kept author as Sunday Esumike, PhD")
print(path)
