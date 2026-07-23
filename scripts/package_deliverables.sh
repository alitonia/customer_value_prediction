#!/usr/bin/env bash
# Package all deliverables into a single zip file.
# Converts .md → .pdf and .pptx → .pdf, copies code/data/model/plots.
#
# Usage: bash scripts/package_deliverables.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="$ROOT/deliverables"
ZIP="$ROOT/customer_value_prediction_deliverables.zip"

# Activate venv for markdown conversion
source .venv/bin/activate

echo "=== Packaging Deliverables ==="
echo ""

# Clean previous run
rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"/{docs,src,app,monitoring,tests,notebooks/plots,data/synthetic,data/processed,models}

# ---------------------------------------------------------------------------
# 1. Convert markdown docs to PDF
# ---------------------------------------------------------------------------
echo "[1/6] Converting markdown docs to PDF..."

convert_md_to_pdf() {
    local md_file="$1"
    local pdf_file="$2"

    python3 -c "
import markdown
from weasyprint import HTML
with open('$md_file') as f:
    md = f.read()
html = markdown.markdown(md, extensions=['tables', 'fenced_code', 'toc'])
full = '''<html><head><meta charset=\"utf-8\"><style>
body { font-family: 'DejaVu Sans', Arial, sans-serif; margin: 40px auto; line-height: 1.7; color: #222; max-width: 850px; font-size: 11pt; }
h1 { color: #1a1a2e; border-bottom: 2px solid #0096d6; padding-bottom: 8px; font-size: 18pt; }
h2 { color: #1a1a2e; margin-top: 28px; border-bottom: 1px solid #ccc; padding-bottom: 4px; font-size: 14pt; }
h3 { color: #333; font-size: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 6px 10px; text-align: left; }
th { background: #1a1a2e; color: white; }
tr:nth-child(even) { background: #f5f5f5; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f0f0f0; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 9pt; }
blockquote { border-left: 3px solid #0096d6; margin: 12px 0; padding: 8px 16px; background: #f0f8ff; }
hr { border: none; border-top: 1px solid #ddd; margin: 24px 0; }
strong { color: #1a1a2e; }
</style></head><body>''' + html + '</body></html>'
HTML(string=full).write_pdf('$pdf_file')
print('  ✓ $(basename "$pdf_file")')
" 2>/dev/null || echo "  ✗ FAILED: $(basename "$pdf_file")"
}

# Convert all narrative docs
for md in docs/data_dictionary.md docs/synthetic_data_schema.md \
          docs/demo_walkthrough.md; do
    if [ -f "$md" ]; then
        convert_md_to_pdf "$md" "$OUT/docs/$(basename "${md%.md}.pdf")"
    fi
done

# Convert generated docs from data/processed
for md in data/processed/cleaning_log.md data/processed/feature_catalog.md data/processed/model_evaluation.md; do
    if [ -f "$md" ]; then
        convert_md_to_pdf "$md" "$OUT/docs/$(basename "${md%.md}.pdf")"
    fi
done

# Convert monitoring spec
if [ -f monitoring/monitoring_spec.md ]; then
    convert_md_to_pdf "monitoring/monitoring_spec.md" "$OUT/docs/monitoring_spec.pdf"
fi

# ---------------------------------------------------------------------------
# 2. Copy slides (PPTX -> PDF already exists, copy both) and final report
# ---------------------------------------------------------------------------
echo "[2/6] Copying slides and report..."
cp docs/slides.pdf "$OUT/docs/" 2>/dev/null && echo "  ✓ slides.pdf"
cp docs/slides.pptx "$OUT/docs/" 2>/dev/null && echo "  ✓ slides.pptx (editable)"
cp docs/final_report.pdf "$OUT/docs/" 2>/dev/null && echo "  ✓ final_report.pdf"

# Copy original project announcement PDF
cp docs/Ecommerce_Order_Value_Prediction_Project_Announcement.pdf "$OUT/docs/" 2>/dev/null && echo "  ✓ project_announcement.pdf"

# ---------------------------------------------------------------------------
# 3. Copy source code
# ---------------------------------------------------------------------------
echo "[3/6] Copying source code..."
cp -r src/data/*.py "$OUT/src/" 2>/dev/null
cp -r src/features/*.py "$OUT/src/" 2>/dev/null
cp -r src/models/*.py "$OUT/src/" 2>/dev/null
cp -r src/common/*.py "$OUT/src/" 2>/dev/null
cp src/config.py "$OUT/src/" 2>/dev/null
cp -r app/api/*.py "$OUT/app/" 2>/dev/null
cp -r app/frontend/*.py "$OUT/app/" 2>/dev/null
cp app.py "$OUT/app/" 2>/dev/null
cp -r monitoring/*.py "$OUT/monitoring/" 2>/dev/null
cp -r tests/*.py "$OUT/tests/" 2>/dev/null
cp run_pipeline.sh "$OUT/" 2>/dev/null
cp Dockerfile "$OUT/" 2>/dev/null
cp requirements.txt "$OUT/" 2>/dev/null
cp README.md "$OUT/" 2>/dev/null
echo "  ✓ src/, app/, monitoring/, tests/, run_pipeline.sh, Dockerfile"

# ---------------------------------------------------------------------------
# 4. Copy notebooks + plots
# ---------------------------------------------------------------------------
echo "[4/6] Copying notebooks and plots..."
cp notebooks/*.ipynb "$OUT/notebooks/" 2>/dev/null
cp notebooks/plots/*.png "$OUT/notebooks/plots/" 2>/dev/null
echo "  ✓ 4 notebooks + 9 plots"

# ---------------------------------------------------------------------------
# 5. Copy data + model + monitoring reports
# ---------------------------------------------------------------------------
echo "[5/6] Copying data, model, and monitoring reports..."
cp data/synthetic/*.parquet "$OUT/data/synthetic/" 2>/dev/null
cp data/processed/*.parquet "$OUT/data/processed/" 2>/dev/null
cp data/processed/*.joblib "$OUT/data/processed/" 2>/dev/null
cp models/*.joblib "$OUT/models/" 2>/dev/null
cp monitoring/evidently_*.html "$OUT/monitoring/" 2>/dev/null
cp monitoring/drift_report.json "$OUT/monitoring/" 2>/dev/null
echo "  ✓ parquet data, model artifact, Evidently dashboards"

# ---------------------------------------------------------------------------
# 6. Create zip
# ---------------------------------------------------------------------------
echo "[6/6] Creating zip archive..."
cd "$OUT"
zip -r "$ZIP" . -x "*.DS_Store" > /dev/null
cd "$ROOT"

SIZE=$(du -sh "$ZIP" | cut -f1)
FILE_COUNT=$(find "$OUT" -type f | wc -l)

echo ""
echo "=== Done ==="
echo "  Deliverables folder: $OUT ($FILE_COUNT files)"
echo "  Zip archive:         $ZIP ($SIZE)"
echo ""
echo "Contents:"
find "$OUT" -type f | sed "s|$OUT/||" | sort | head -60
if [ "$FILE_COUNT" -gt 60 ]; then
    echo "  ... and $((FILE_COUNT - 60)) more files"
fi
