import sys
import os
from fastapi.testclient import TestClient

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "cannabinoids-testing-api"}
    print("Health check endpoint test PASSED!")

def test_analyze_endpoint():
    pos_msp_path = os.path.join(project_root, "known_compound_recognize", "positive.msp")
    with open(pos_msp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = content.strip().split("\n\n")
    sample_block = blocks[0] + "\n\n"
    
    files = {
        "file": ("test_sample.msp", sample_block.encode('utf-8'), "text/plain")
    }
    
    response = client.post("/api/v1/analyze?min_similarity=0.75", files=files)
    assert response.status_code == 200, f"Error: {response.text}"
    json_data = response.json()
    assert json_data["filename"] == "test_sample.msp"
    assert json_data["known_library_match"]["is_matched"] is True
    assert json_data["model_inference"]["is_high_risk"] is True
    print("Analyze endpoint test PASSED!")

if __name__ == "__main__":
    test_health()
    test_analyze_endpoint()
