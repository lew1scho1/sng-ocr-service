from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import asyncio
from datetime import datetime

from .ocr_service import process_image, extract_barcodes

app = FastAPI(
    title="SNG OCR Service",
    description="인보이스 이미지에서 바코드를 추출하는 OCR 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (Production에서는 Redis 사용 권장)
jobs: dict = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: str
    completed_at: Optional[str] = None
    barcodes: Optional[list] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None


class OcrRequest(BaseModel):
    image_url: str
    callback_url: Optional[str] = None


@app.get("/")
async def root():
    return {"service": "SNG OCR Service", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/ocr/jobs", response_model=JobStatus)
async def create_ocr_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    OCR 작업 생성 (이미지 파일 업로드)
    """
    job_id = str(uuid.uuid4())

    # 파일 읽기
    image_data = await file.read()

    # Job 생성
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "barcodes": None,
        "raw_text": None,
        "error": None
    }

    # 백그라운드에서 OCR 처리
    background_tasks.add_task(process_ocr_job, job_id, image_data)

    return JobStatus(**jobs[job_id])


@app.post("/api/v1/ocr/jobs/sync")
async def create_ocr_job_sync(file: UploadFile = File(...)):
    """
    OCR 작업 (동기 처리 - 즉시 결과 반환)
    """
    try:
        image_data = await file.read()

        # OCR 처리
        raw_text = process_image(image_data)
        barcodes = extract_barcodes(raw_text)

        return {
            "success": True,
            "barcodes": barcodes,
            "barcode_count": len(barcodes),
            "raw_text_preview": raw_text[:500] if raw_text else ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ocr/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    OCR 작업 상태 조회
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**jobs[job_id])


async def process_ocr_job(job_id: str, image_data: bytes):
    """
    백그라운드에서 OCR 처리
    """
    try:
        jobs[job_id]["status"] = "processing"

        # OCR 처리
        raw_text = process_image(image_data)
        barcodes = extract_barcodes(raw_text)

        jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "barcodes": barcodes,
            "raw_text": raw_text[:1000] if raw_text else ""  # 1000자 제한
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e)
        })


# Job cleanup (오래된 작업 삭제)
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_jobs())


async def cleanup_old_jobs():
    """30분마다 1시간 이상 된 작업 삭제"""
    while True:
        await asyncio.sleep(1800)  # 30분
        now = datetime.utcnow()
        to_delete = []

        for job_id, job in jobs.items():
            created = datetime.fromisoformat(job["created_at"])
            if (now - created).total_seconds() > 3600:  # 1시간
                to_delete.append(job_id)

        for job_id in to_delete:
            del jobs[job_id]
