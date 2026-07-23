import os
import subprocess
import sys


def main():
    streamlit_enable = os.environ.get("STREAMLIT_ENABLE", "").strip().lower()
    
    if streamlit_enable in ["true", "1", "yes"]:
        print("Starting Streamlit UI on port 8001...")
        cmd = [
            "streamlit", "run", "streamlit/app.py",
            "--server.port", "8001",
            "--server.address", "0.0.0.0"
        ]
    else:
        print("Starting FastAPI server on port 8000...")
        cmd = [
            "uvicorn", "server:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        
    # On Unix-like systems, execvp replaces the current process
    if sys.platform != "win32":
        try:
            os.execvp(cmd[0], cmd)
        except FileNotFoundError:
            # Fallback if command not directly in PATH (e.g. running outside uv environment)
            # Try running with python -m or subprocess
            pass

    # Fallback/Windows run
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
