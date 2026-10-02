from pathlib import Path
import re

dashboard = Path("src/18_streamlit_dashboard.py")

if not dashboard.exists():
    raise FileNotFoundError("Dashboard file not found: src/18_streamlit_dashboard.py")

text = dashboard.read_text(encoding="utf-8")

# Remove Final Report from sidebar menu.
text = text.replace(
'''        "Upload Metrics Explorer",
        "Final Report"''',
'''        "Upload Metrics Explorer"'''
)

# Remove Final Report page block if present.
final_report_pattern = r'''# ------------------------------------------------------------
# Final Report
# ------------------------------------------------------------
elif page == "Final Report":
.*?\Z'''

text = re.sub(final_report_pattern, "", text, flags=re.S)

# Add dark CSS after st.title if not already added.
dark_css = '''
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
'''

if "background-color: #0e1117" not in text:
    text = text.replace(
        'st.title("Adversarial IDS Interactive Dashboard")',
        'st.title("Adversarial IDS Interactive Dashboard")\n' + dark_css
    )

dashboard.write_text(text, encoding="utf-8")

# Create Streamlit theme config.
config_dir = Path(".streamlit")
config_dir.mkdir(exist_ok=True)

config = Path(".streamlit/config.toml")
config.write_text(
'''[theme]
base = "dark"
primaryColor = "#60a5fa"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#111827"
textColor = "#f8fafc"
font = "sans serif"

[server]
headless = false
''',
encoding="utf-8"
)

print("Dashboard updated.")
print("- Final Report page removed")
print("- Dark theme config created")
print("- Dark CSS added")
print("Files updated:")
print(" ", dashboard)
print(" ", config)
