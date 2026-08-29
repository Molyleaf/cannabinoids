import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from app.i18n import get_text, set_language

test_strings = [
    "NPS Spectral Intelligence Platform",
    "Analysis Workflow Instructions",
    "1. Select Model: binary model (SC versus NSC) or multi-class model (nine NPS categories).",
    "2. Sign the data sharing consent: consent to share spectrum data or not.",
    "3. Upload File: Supports .msp and .mgf formats.",
    "Binary Model (SC versus NSC)",
    "Multi-class Model (nine NPS categories)",
    "Library Match Threshold (Similarity)",
    "Below 0.80 Threshold",
    "Fentanyls",
    "Cathinones",
    "Synthetic Cannabinoids",
    "Arylcyclohexylamines",
    "Benzodiazepines",
    "Nitazenes",
    "Opiates",
    "Phenethylamines",
    "Tryptamines",
    "Positive (High Risk)",
    "Negative (Low Risk)",
    "Pending Manual Review",
]

print("=== Testing en_US (English default) ===")
set_language("en_US")
for s in test_strings:
    print(f"EN: '{s}' -> '{get_text(s)}'")

print("\n=== Testing zh_CN (Chinese translation) ===")
set_language("zh_CN")
for s in test_strings:
    print(f"ZH: '{s}' -> '{get_text(s)}'")


