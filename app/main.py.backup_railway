from pathlib import Path
import re
import sys
from datetime import datetime

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("app/main.py")

if not target.exists():
    raise SystemExit(f"Could not find {target}. Run this from the backend folder or pass the path to main.py.")

text = target.read_text(encoding="utf-8")

backup = target.with_suffix(target.suffix + ".backup_railway")
backup.write_text(text, encoding="utf-8")

changes = []

def replace_once(old, new, label):
    global text
    if old in text:
        text = text.replace(old, new, 1)
        changes.append(label)

def replace_all(old, new, label):
    global text
    count = text.count(old)
    if count:
        text = text.replace(old, new)
        changes.append(f"{label} ({count})")

heavy_imports = [
    "from reportlab.lib.pagesizes import A4\n",
    "from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle\n",
    "from reportlab.lib.styles import getSampleStyleSheet\n",
    "from reportlab.lib import colors\n",
    "from reportlab.platypus import Image as RLImage\n",
    "from reportlab.lib.units import inch\n",
    "from playwright.sync_api import sync_playwright\n",
    "from app.services.scan_validator import validate_scan_image\n",
]

for line in heavy_imports:
    if line in text:
        text = text.replace(line, "")
        changes.append(f"removed startup import: {line.strip()}")

text2 = re.sub(
    r"(?m)^from app\.services\.image_prediction import predict_image\s*$",
    "# from app.services.image_prediction import predict_image",
    text,
)
if text2 != text:
    text = text2
    changes.append("disabled TensorFlow startup import")

if "HTTPException" not in text.split("\n", 5)[0:6]:
    text = text.replace(
        "from fastapi import FastAPI\n",
        "from fastapi import FastAPI, Header, HTTPException\n",
        1,
    )

replace_once(
    "    try:\n        scan_check = validate_scan_image(image_path)\n",
    "    try:\n        from app.services.scan_validator import validate_scan_image\n\n        scan_check = validate_scan_image(image_path)\n",
    "lazy-loaded scan validator",
)

if "PDF generator is not available on this deployment" not in text:
    old = """def generate_pdf_report(
    diagnosis_id: int,
    download: bool = False,
    db: Session = Depends(get_db),
):
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
"""
    new = """def generate_pdf_report(
    diagnosis_id: int,
    download: bool = False,
    db: Session = Depends(get_db),
):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import inch
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"PDF generator is not available on this deployment: {str(e)}",
        )

    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
"""
    if old in text:
        text = text.replace(old, new, 1)
        changes.append("lazy-loaded ReportLab PDF generator")
    else:
        pattern = re.compile(
            r"(def generate_pdf_report\(\n(?:.*\n)*?\):\n)(\s+diagnosis = db\.query\(Diagnosis\)\.filter\(Diagnosis\.id == diagnosis_id\)\.first\(\)\n)",
            re.MULTILINE,
        )
        m = pattern.search(text)
        if m:
            imports_block = """    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import inch
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"PDF generator is not available on this deployment: {str(e)}",
        )

"""
            text = text[:m.start(2)] + imports_block + text[m.start(2):]
            changes.append("lazy-loaded ReportLab PDF generator")

if "HTML PDF generator is not available on this deployment" not in text:
    replace_once(
        "    with sync_playwright() as p:\n",
        """    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"HTML PDF generator is not available on this deployment: {str(e)}",
        )

    with sync_playwright() as p:
""",
        "lazy-loaded Playwright PDF generator",
    )

replace_all(
    'http://127.0.0.1:8000/{route}/{escape(filename)}',
    '/{route}/{escape(filename)}',
    "fixed report image URLs",
)
replace_all(
    'src="http://127.0.0.1:8000/uploads/{original_filename}"',
    'src="/uploads/{original_filename}"',
    "fixed original scan report URL",
)
replace_all(
    'src="http://127.0.0.1:8000/heatmaps/{heatmap_filename}"',
    'src="/heatmaps/{heatmap_filename}"',
    "fixed heatmap report URL",
)

target.write_text(text, encoding="utf-8")

print("Done. Backup created:", backup)
if changes:
    print("Changes:")
    for item in changes:
        print("-", item)
else:
    print("No changes were needed.")
