from pathlib import Path
import json
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from xgboost import XGBClassifier


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Adversarial IDS Interactive Dashboard",
    page_icon="🛡️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = MODEL_DIR / "baseline"
DEFENSE_DIR = MODEL_DIR / "defense"


# ------------------------------------------------------------
# Model classes
# ------------------------------------------------------------
class IDSClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x.float())


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def pct(x):
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"


def safe_read_json(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def file_status(path):
    return "FOUND" if path.exists() else "MISSING"


@st.cache_resource
def load_pytorch_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    input_dim = int(checkpoint["input_dim"])
    model = IDSClassifier(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_dim


@st.cache_resource
def load_xgboost_model(path):
    model = XGBClassifier()
    model.load_model(path)
    return model


@st.cache_resource
def load_joblib(path):
    return joblib.load(path)


def predict_torch(model, X, batch_size=4096):
    preds = []
    scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            scores.append(probs[:, 1].cpu().numpy())

    return np.concatenate(preds), np.concatenate(scores)


def align_uploaded_features(df, feature_cols):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in feature_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_cols and c not in ["Label", "binary_label", "attack_label", "source_file"]]

    if missing:
        return None, missing, extra

    X_raw = df[feature_cols].copy()

    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    return X_raw, missing, extra


def run_prediction_pipeline(uploaded_df, feature_cols, imputer, scaler, pytorch_model, xgb_model, defense_model):
    X_raw, missing, extra = align_uploaded_features(uploaded_df, feature_cols)

    if missing:
        return None, missing, extra

    X_imp = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imp)
    X_scaled = np.clip(X_scaled, 0.0, 1.0).astype(np.float32)

    pytorch_pred, pytorch_score = predict_torch(pytorch_model, X_scaled)
    xgb_score = xgb_model.predict_proba(X_scaled)[:, 1]
    xgb_pred = (xgb_score >= 0.5).astype(int)
    defense_pred, defense_score = predict_torch(defense_model, X_scaled)

    output = uploaded_df.copy()
    output["pytorch_prediction"] = np.where(pytorch_pred == 1, "ATTACK", "BENIGN")
    output["pytorch_attack_score"] = pytorch_score
    output["xgboost_prediction"] = np.where(xgb_pred == 1, "ATTACK", "BENIGN")
    output["xgboost_attack_score"] = xgb_score
    output["defense_prediction"] = np.where(defense_pred == 1, "ATTACK", "BENIGN")
    output["defense_attack_score"] = defense_score

    return output, missing, extra


# ------------------------------------------------------------
# Load project artifacts
# ------------------------------------------------------------
baseline_metrics = safe_read_json(METRICS_DIR / "baseline_metrics.json")
feasibility = safe_read_csv(METRICS_DIR / "feature_feasibility_table.csv")
phase3 = safe_read_csv(METRICS_DIR / "phase3_whitebox_full_summary.csv")
phase4a = safe_read_csv(METRICS_DIR / "phase4a_transfer_full_summary.csv")
phase4b = safe_read_csv(METRICS_DIR / "phase4b_hopskipjump_full_summary.csv")
comparison = safe_read_csv(METRICS_DIR / "final_defense_comparison_table.csv")
feature_squeezing = safe_read_json(METRICS_DIR / "phase5a_feature_squeezing_summary.json")
adversarial_training = safe_read_json(METRICS_DIR / "phase5b_adversarial_training_summary.json")

feature_path = BASELINE_DIR / "feature_columns.json"
feature_cols = safe_read_json(feature_path)

report_path = REPORT_DIR / "final_report.md"
report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else "Final report not found."


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Dashboard Controls")

page = st.sidebar.radio(
    "Choose workspace",
    [
        "Project Health",
        "About Project",
        "Interactive Prediction Lab",
        "Attack Explorer",
        "Defense Explorer",
        "Upload Metrics Explorer"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Author")
st.sidebar.markdown("**Sunday Esumike, PhD**")

st.sidebar.caption("Project root")
st.sidebar.code(str(PROJECT_ROOT))


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Adversarial IDS Interactive Dashboard")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    h1, h2, h3, h4 {
        color: #f8fafc;
    }

    p, li, label, span, div {
        color: #e5e7eb;
    }

    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        border: 1px solid #60a5fa;
    }

    .stDownloadButton > button {
        background-color: #059669;
        color: white;
        border-radius: 10px;
        border: 1px solid #34d399;
    }

    code {
        color: #93c5fd;
    }
</style>
""", unsafe_allow_html=True)

st.caption("Load traffic data, inspect project results, compare baseline and defended IDS behavior, and explore adversarial attack outcomes.")


# ------------------------------------------------------------
# Project Health
# ------------------------------------------------------------
if page == "Project Health":
    st.header("Project Health Check")

    required_files = {
        "Baseline metrics": METRICS_DIR / "baseline_metrics.json",
        "Feature feasibility table": METRICS_DIR / "feature_feasibility_table.csv",
        "White-box attack summary": METRICS_DIR / "phase3_whitebox_full_summary.csv",
        "Transfer attack summary": METRICS_DIR / "phase4a_transfer_full_summary.csv",
        "HopSkipJump summary": METRICS_DIR / "phase4b_hopskipjump_full_summary.csv",
        "Defense comparison": METRICS_DIR / "final_defense_comparison_table.csv",
        "PyTorch baseline": BASELINE_DIR / "pytorch_clean_baseline.pt",
        "XGBoost baseline": BASELINE_DIR / "xgboost_clean_baseline.json",
        "Median imputer": BASELINE_DIR / "median_imputer.joblib",
        "MinMax scaler": BASELINE_DIR / "minmax_scaler.joblib",
        "Adversarial training defense": DEFENSE_DIR / "pytorch_adversarial_training_defense.pt",
        "Final report": REPORT_DIR / "final_report.md",
    }

    health = pd.DataFrame([
        {"artifact": name, "status": file_status(path), "path": str(path)}
        for name, path in required_files.items()
    ])

    st.dataframe(health, use_container_width=True)

    found_count = (health["status"] == "FOUND").sum()
    st.metric("Artifacts Found", f"{found_count} / {len(health)}")

    if baseline_metrics:
        st.subheader("Baseline Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input Features", baseline_metrics.get("input_dim", "N/A"))
        c2.metric("Train Rows", baseline_metrics.get("train_rows", "N/A"))
        c3.metric("Validation Rows", baseline_metrics.get("validation_rows", "N/A"))
        c4.metric("Test Rows", baseline_metrics.get("test_rows", "N/A"))


# ------------------------------------------------------------
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




# ------------------------------------------------------------
# Interactive Prediction Lab
# ------------------------------------------------------------
elif page == "Interactive Prediction Lab":
    st.header("Interactive Prediction Lab")

    st.markdown("""
    Upload a traffic-flow CSV that contains the same feature columns used by the trained IDS.
    The dashboard will apply the frozen imputer and scaler, then run predictions through:

    - PyTorch baseline IDS
    - XGBoost baseline IDS
    - Adversarial-training defense IDS
    """)

    needed = [
        feature_cols,
        BASELINE_DIR / "median_imputer.joblib",
        BASELINE_DIR / "minmax_scaler.joblib",
        BASELINE_DIR / "pytorch_clean_baseline.pt",
        BASELINE_DIR / "xgboost_clean_baseline.json",
        DEFENSE_DIR / "pytorch_adversarial_training_defense.pt",
    ]

    if feature_cols is None:
        st.error("Missing feature column file. Cannot run prediction lab.")
        st.stop()

    uploaded = st.file_uploader(
        "Upload traffic CSV",
        type=["csv"],
        help="Use a CSV with the same feature columns as the model training data."
    )

    with st.expander("Required feature columns"):
        st.write(f"Required feature count: {len(feature_cols)}")
        st.dataframe(pd.DataFrame({"feature": feature_cols}), use_container_width=True)

    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded, low_memory=False)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            st.stop()

        st.subheader("Uploaded Data Preview")
        st.write(f"Rows: {len(uploaded_df):,} | Columns: {len(uploaded_df.columns):,}")
        st.dataframe(uploaded_df.head(50), use_container_width=True)

        imputer = load_joblib(BASELINE_DIR / "median_imputer.joblib")
        scaler = load_joblib(BASELINE_DIR / "minmax_scaler.joblib")
        pytorch_model, _ = load_pytorch_model(BASELINE_DIR / "pytorch_clean_baseline.pt")
        xgb_model = load_xgboost_model(BASELINE_DIR / "xgboost_clean_baseline.json")
        defense_model, _ = load_pytorch_model(DEFENSE_DIR / "pytorch_adversarial_training_defense.pt")

        max_rows = st.slider(
            "Rows to score",
            min_value=1,
            max_value=min(len(uploaded_df), 10000),
            value=min(len(uploaded_df), 1000),
            step=100 if len(uploaded_df) >= 100 else 1
        )

        run_button = st.button("Run IDS Predictions", type="primary")

        if run_button:
            score_df = uploaded_df.head(max_rows).copy()

            results, missing, extra = run_prediction_pipeline(
                score_df,
                feature_cols,
                imputer,
                scaler,
                pytorch_model,
                xgb_model,
                defense_model
            )

            if missing:
                st.error("Uploaded CSV is missing required model features.")
                st.write("Missing columns:")
                st.dataframe(pd.DataFrame({"missing_feature": missing}), use_container_width=True)
                if extra:
                    st.warning("The CSV also has extra columns that are not model inputs.")
                    st.dataframe(pd.DataFrame({"extra_column": extra[:100]}), use_container_width=True)
                st.stop()

            st.success("Predictions complete.")

            c1, c2, c3 = st.columns(3)
            c1.metric("PyTorch ATTACK rows", int((results["pytorch_prediction"] == "ATTACK").sum()))
            c2.metric("XGBoost ATTACK rows", int((results["xgboost_prediction"] == "ATTACK").sum()))
            c3.metric("Defense ATTACK rows", int((results["defense_prediction"] == "ATTACK").sum()))

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            agreement = pd.crosstab(
                results["pytorch_prediction"],
                results["defense_prediction"],
                rownames=["PyTorch baseline"],
                colnames=["Defense model"]
            )

            st.subheader("Baseline vs Defense Agreement")
            st.dataframe(agreement, use_container_width=True)

            output_csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download prediction results CSV",
                data=output_csv,
                file_name="ids_prediction_results.csv",
                mime="text/csv"
            )


# ------------------------------------------------------------
# Attack Explorer
# ------------------------------------------------------------
elif page == "Attack Explorer":
    st.header("Attack Explorer")

    if phase3 is None or phase4a is None or phase4b is None:
        st.error("One or more attack result files are missing.")
        st.stop()

    attack_family = st.selectbox(
        "Select attack result type",
        ["White-box FGSM/PGD", "Black-box transfer", "HopSkipJump"]
    )

    if attack_family == "White-box FGSM/PGD":
        method = st.selectbox("Attack method", sorted(phase3["attack_method"].unique()))
        filtered = phase3[phase3["attack_method"] == method]

        st.subheader(f"{method} Evasion by Epsilon")
        chart = filtered.pivot(index="epsilon", columns="constraint_mode", values="evasion_rate")
        st.line_chart(chart)

        st.dataframe(filtered, use_container_width=True)

    elif attack_family == "Black-box transfer":
        st.subheader("Transfer Evasion by Epsilon")
        chart = phase4a.set_index("epsilon")[
            [
                "surrogate_whitebox_evasion_rate",
                "pytorch_transfer_evasion_rate",
                "xgboost_transfer_evasion_rate"
            ]
        ]
        st.line_chart(chart)
        st.dataframe(phase4a, use_container_width=True)

    else:
        st.subheader("HopSkipJump Raw vs Projected Evasion")
        chart = phase4b.set_index("epsilon_projection")[
            ["raw_unprojected_evasion_rate", "constrained_projected_evasion_rate"]
        ]
        st.line_chart(chart)
        st.dataframe(phase4b, use_container_width=True)


# ------------------------------------------------------------
# Defense Explorer
# ------------------------------------------------------------
elif page == "Defense Explorer":
    st.header("Defense Explorer")

    if comparison is None:
        st.error("Defense comparison table is missing.")
        st.stop()

    st.subheader("Final Defense Comparison")
    st.dataframe(comparison, use_container_width=True)

    metric_options = [
        "clean_accuracy",
        "clean_attack_recall",
        "false_positive_rate",
        "fgsm_constrained_evasion_eps_0_100",
        "pgd_constrained_evasion_eps_0_100"
    ]

    selected_metrics = st.multiselect(
        "Choose metrics to visualize",
        metric_options,
        default=[
            "clean_attack_recall",
            "fgsm_constrained_evasion_eps_0_100",
            "pgd_constrained_evasion_eps_0_100"
        ]
    )

    if selected_metrics:
        chart = comparison.set_index("scenario")[selected_metrics]
        st.bar_chart(chart)

    if feature_squeezing:
        with st.expander("Feature squeezing clean metrics"):
            st.json(feature_squeezing.get("clean_metrics_after_squeezing", {}))

    if adversarial_training:
        with st.expander("Adversarial training clean metrics"):
            st.json(adversarial_training.get("clean_metrics_after_defense", {}))


# ------------------------------------------------------------
# Upload Metrics Explorer
# ------------------------------------------------------------
elif page == "Upload Metrics Explorer":
    st.header("Upload Metrics Explorer")

    st.markdown("""
    Upload any project CSV or JSON output file. The dashboard will display it and create simple charts
    when numeric columns are available.
    """)

    uploaded_metric = st.file_uploader(
        "Upload metrics CSV or JSON",
        type=["csv", "json"]
    )

    if uploaded_metric is not None:
        name = uploaded_metric.name.lower()

        try:
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_metric)
                st.success("CSV loaded.")
                st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
                st.dataframe(df, use_container_width=True)

                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

                if numeric_cols:
                    x_col = st.selectbox("X-axis", df.columns.tolist())
                    y_cols = st.multiselect("Y-axis numeric columns", numeric_cols, default=numeric_cols[:2])

                    if y_cols:
                        chart_df = df.set_index(x_col)[y_cols]
                        st.line_chart(chart_df)

            else:
                content = uploaded_metric.read().decode("utf-8")
                obj = json.loads(content)
                st.success("JSON loaded.")
                st.json(obj)

        except Exception as e:
            st.error(f"Could not parse uploaded metrics file: {e}")


