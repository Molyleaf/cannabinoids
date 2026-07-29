# -*- coding: utf-8 -*-
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "streamlit" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8001",
        "--server.address=0.0.0.0",
        "--server.baseUrlPath=cannabinoids",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
        "--server.corsAllowedOrigins=[\"https://chem.warships.cn\", \"https://mspredict.com\"]"
    ]
    print(f"Starting Streamlit server on http://0.0.0.0:8001/cannabinoids (app: {app_path})...")
    subprocess.run(cmd)
