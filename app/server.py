from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from app.pipeline import run_pipeline

app = FastAPI(
    title="Cannabinoids Spectrum Risk Analysis API",
    description="FastAPI 服务器：结合 ms_entropy 库检索与 safetensors 深度学习模型的合成大麻素质谱测样服务",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LibraryMatchResponse(BaseModel):
    is_matched: bool
    matched_smiles: Optional[str] = ""
    min_similarity_threshold: float

class ModelInferenceResponse(BaseModel):
    risk_probability: float
    risk_percentage: str
    risk_level: str
    is_high_risk: bool

class AnalysisResponse(BaseModel):
    filename: str
    num_cleaned_peaks: int
    precursor_mz: Optional[float] = None
    known_library_match: LibraryMatchResponse
    model_inference: ModelInferenceResponse
    peaks: List[List[float]]

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "cannabinoids-testing-api"}

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_spectrum_endpoint(
    file: UploadFile = File(...),
    min_similarity: float = Query(0.75, ge=0.0, le=1.0, description="已知库特征比对的最小相似度阈值")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ["msp", "mgf"]:
        raise HTTPException(status_code=400, detail="Only .msp or .mgf files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = run_pipeline(
            file_bytes=content,
            filename=file.filename,
            min_similarity=min_similarity
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")
