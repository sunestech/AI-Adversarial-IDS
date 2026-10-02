from pathlib import Path
import win32com.client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCX = PROJECT_ROOT / "reports" / "final_report.docx"
PDF = PROJECT_ROOT / "reports" / "final_report.pdf"

if not DOCX.exists():
    raise FileNotFoundError(f"Missing DOCX file: {DOCX}")

word = win32com.client.Dispatch("Word.Application")
word.Visible = False

try:
    doc = word.Documents.Open(str(DOCX))
    doc.SaveAs(str(PDF), FileFormat=17)
    doc.Close()
finally:
    word.Quit()

print("PDF report saved:")
print(PDF)
