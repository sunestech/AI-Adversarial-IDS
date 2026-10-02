from pathlib import Path

path = Path("src/18_streamlit_dashboard.py")

if not path.exists():
    raise FileNotFoundError("Dashboard file not found: src/18_streamlit_dashboard.py")

text = path.read_text(encoding="utf-8")

# Add About Project to the sidebar page list.
text = text.replace(
'''        "Project Health",
        "Interactive Prediction Lab",''',
'''        "Project Health",
        "About Project",
        "Interactive Prediction Lab",'''
)

# Add author block to the sidebar.
text = text.replace(
'''st.sidebar.caption("Project root")
st.sidebar.code(str(PROJECT_ROOT))''',
'''st.sidebar.caption("Author")
st.sidebar.markdown("**Sunday Esumike, PhD**")

st.sidebar.caption("Project root")
st.sidebar.code(str(PROJECT_ROOT))'''
)

# Insert About Project page before Interactive Prediction Lab.
insert_before = '''# ------------------------------------------------------------
# Interactive Prediction Lab
# ------------------------------------------------------------'''

about_page = '''# ------------------------------------------------------------
# About Project
# ------------------------------------------------------------
elif page == "About Project":
    st.header("About This Project")

    st.markdown("""
    **Author:** Sunday Esumike, PhD

    **Project Title:** Adversarial Machine Learning Assessment of an Intrusion Detection System

    This project builds a machine-learning Intrusion Detection System, evaluates how vulnerable it is to adversarial evasion, applies realistic network-flow feasibility constraints, tests black-box attack behavior, and evaluates defensive controls.

    The project is designed as a security-research and red-team style assessment, not just a standard classification model.
    """)

    st.subheader("Research Question")

    st.info(
        "How vulnerable is a machine-learning IDS to adversarial evasion, how much of that vulnerability survives realistic network-flow constraints, can attacks transfer in black-box conditions, and how much can defensive techniques reduce that risk?"
    )

    st.subheader("Project Workflow")

    workflow = pd.DataFrame([
        {"Phase": "Phase 0", "Name": "Environment and Data Audit", "Purpose": "Verify resources, inspect ZIP files, combine and clean IDS flow data."},
        {"Phase": "Phase 1", "Name": "Baseline IDS Models", "Purpose": "Train PyTorch and XGBoost IDS baselines and freeze artifacts."},
        {"Phase": "Phase 2", "Name": "Feature Feasibility", "Purpose": "Decide which network-flow features an attacker can realistically perturb."},
        {"Phase": "Phase 3", "Name": "White-Box Attacks", "Purpose": "Run FGSM and PGD attacks in unconstrained and constrained modes."},
        {"Phase": "Phase 4A", "Name": "Transfer Attack", "Purpose": "Train a surrogate model and test black-box transfer evasion."},
        {"Phase": "Phase 4B", "Name": "HopSkipJump", "Purpose": "Run decision-based black-box attacks on a bounded subset."},
        {"Phase": "Phase 5", "Name": "Defenses", "Purpose": "Evaluate feature squeezing and adversarial training defenses."},
        {"Phase": "Phase 6", "Name": "Final Report", "Purpose": "Generate a red-team style security assessment report."},
        {"Phase": "Dashboard", "Name": "Interactive Demo", "Purpose": "Load data, run IDS predictions, and explore attack and defense results interactively."},
    ])

    st.dataframe(workflow, use_container_width=True)

    st.subheader("Core Finding")

    st.markdown("""
    The major finding is that **unconstrained adversarial attacks can produce very high evasion rates**, but those numbers can be misleading because many feature changes may not represent valid network traffic.

    When perturbations are restricted to a conservative set of attacker-controllable network-flow features, evasion becomes lower and more realistic. Adversarial training produced the strongest tested defense.
    """)

    st.subheader("Key Deliverables")

    deliverables = pd.DataFrame([
        {"Deliverable": "Final report", "Path": "reports/final_report.md"},
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

    st.subheader("How to Use This Dashboard")

    st.markdown("""
    - Use **Project Health** to confirm all expected project artifacts exist.
    - Use **Interactive Prediction Lab** to upload new traffic-flow CSV data and run model predictions.
    - Use **Attack Explorer** to compare constrained and unconstrained adversarial evasion.
    - Use **Defense Explorer** to compare baseline and defended IDS behavior.
    - Use **Upload Metrics Explorer** to load additional CSV or JSON result files.
    - Use **Final Report** to review and download the generated report.
    """)


'''

if insert_before not in text:
    raise ValueError("Could not find insertion point for About Project page.")

text = text.replace(insert_before, about_page + "\n\n" + insert_before)

path.write_text(text, encoding="utf-8")

print("Dashboard updated with About Project page and author name.")
print(path)
