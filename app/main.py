from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
import asyncio
import logging
import threading
import tempfile
import os
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .ocr_service import process_image, extract_barcodes
from .parsers.sng_parser import parse_sng_invoice, to_dict as sng_to_dict

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
    # SNG 전용 필드
    company: Optional[str] = None
    header: Optional[dict] = None
    line_items: Optional[list] = None
    line_item_count: Optional[int] = None


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
    바코드 패턴(12-14자리 숫자)만 추출
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


@app.post("/api/v1/ocr/jobs/sync/sng")
async def create_ocr_job_sync_sng(file: UploadFile = File(...)):
    """
    SNG (Shake-N-Go) 인보이스 전용 OCR (단일 이미지)

    추출 정보:
    - 헤더: Invoice NO, Invoice Date
    - 라인 아이템: ITEM_CODE + COLOR + QUANTITY + UNIT_PRICE
    """
    try:
        image_data = await file.read()

        # OCR 처리
        raw_text = process_image(image_data)

        # 디버그: 전체 OCR 텍스트 로그 출력
        logger.info("=" * 80)
        logger.info("SNG OCR - FULL RAW TEXT START")
        logger.info("=" * 80)
        logger.info(raw_text)
        logger.info("=" * 80)
        logger.info(f"SNG OCR - FULL RAW TEXT END (Total: {len(raw_text)} chars)")
        logger.info("=" * 80)

        # SNG 파서로 구조화된 데이터 추출
        result = parse_sng_invoice(raw_text)

        return {
            "success": True,
            "company": "SNG",
            "header": {
                "invoice_number": result.header.invoice_number,
                "invoice_date": result.header.invoice_date
            },
            "line_items": [
                {
                    "item_code": item.item_code,
                    "color": item.color,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "description": item.description,
                    # DB 매칭용 확장 필드
                    "raw_item_code": item.raw_item_code,
                    "item_code_candidates": item.item_code_candidates,
                    "description_tokens": {
                        "type": item.description_tokens.type if item.description_tokens else None,
                        "length": item.description_tokens.length if item.description_tokens else None,
                        "pcs": item.description_tokens.pcs if item.description_tokens else None,
                        "style": item.description_tokens.style if item.description_tokens else None,
                    } if item.description_tokens else None
                }
                for item in result.line_items
            ],
            "line_item_count": len(result.line_items),
            "raw_text_preview": raw_text[:500] if raw_text else "",
            "last_item_code": result.last_item_code,
            "last_unit_price": result.last_unit_price
        }
    except Exception as e:
        logger.error(f"SNG OCR Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ocr/jobs/sng")
async def create_sng_ocr_job(file: UploadFile = File(...)):
    """
    SNG 인보이스 OCR 작업 생성 (비동기)

    핵심: HTTP 응답과 OCR 처리를 완전히 분리
    1. 파일을 임시 파일로 저장
    2. job_id 생성 및 상태 저장
    3. 별도 스레드에서 OCR 처리 시작
    4. 즉시 응답 반환 (1-2초 내)
    """
    logger.info(f"SNG OCR Job 요청 수신: filename={file.filename}")

    job_id = str(uuid.uuid4())

    # 1. 파일을 임시 파일로 저장 (HTTP 응답 전에 완료해야 함)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_path = temp_file.name

    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        logger.info(f"Job {job_id}: 파일 저장 완료, path={temp_path}, size={len(content)} bytes")
    except Exception as e:
        temp_file.close()
        os.unlink(temp_path)
        logger.error(f"Job {job_id}: 파일 저장 실패 - {str(e)}")
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {str(e)}")

    # 2. Job 생성 (최소 정보만)
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "barcodes": None,
        "raw_text": None,
        "error": None,
        "company": "SNG",
        "header": None,
        "line_items": None,
        "line_item_count": None
    }

    # 3. 별도 스레드에서 OCR 처리 시작 (HTTP 응답과 완전 분리)
    thread = threading.Thread(
        target=process_sng_ocr_job_sync,
        args=(job_id, temp_path),
        daemon=True
    )
    thread.start()

    logger.info(f"Job {job_id}: 스레드 시작, 응답 반환 (status=pending)")

    # 4. 즉시 응답 반환 (최소 payload)
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/v1/ocr/jobs/sync/sng/multi")
async def create_ocr_job_sync_sng_multi(files: List[UploadFile] = File(...)):
    """
    SNG (Shake-N-Go) 인보이스 전용 OCR (멀티 페이지)

    여러 이미지를 순차 처리하여 페이지 간 아이템 연결 지원
    - 페이지 2에서 아이템 코드 없이 색상만 있으면 페이지 1의 마지막 아이템에 연결

    추출 정보:
    - 헤더: Invoice NO, Invoice Date (첫 페이지에서 추출)
    - 라인 아이템: ITEM_CODE + COLOR + QUANTITY + UNIT_PRICE (모든 페이지 병합)
    """
    try:
        all_line_items = []
        header = None
        prev_item_code = None
        prev_unit_price = None
        page_results = []

        for i, file in enumerate(files):
            image_data = await file.read()

            # OCR 처리
            raw_text = process_image(image_data)

            logger.info(f"SNG OCR - Page {i + 1}/{len(files)}, {len(raw_text)} chars")

            # SNG 파서로 구조화된 데이터 추출 (이전 페이지 정보 전달)
            result = parse_sng_invoice(raw_text, prev_item_code, prev_unit_price)

            # 첫 페이지에서 헤더 추출
            if header is None and result.header.invoice_number:
                header = result.header

            # 라인 아이템 병합
            all_line_items.extend(result.line_items)

            # 다음 페이지를 위해 마지막 아이템 정보 저장
            prev_item_code = result.last_item_code
            prev_unit_price = result.last_unit_price

            # 페이지별 결과 저장
            page_results.append({
                "page": i + 1,
                "filename": file.filename,
                "item_count": len(result.line_items),
                "last_item_code": result.last_item_code
            })

        return {
            "success": True,
            "company": "SNG",
            "page_count": len(files),
            "header": {
                "invoice_number": header.invoice_number if header else None,
                "invoice_date": header.invoice_date if header else None
            },
            "line_items": [
                {
                    "item_code": item.item_code,
                    "color": item.color,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "description": item.description,
                    # DB 매칭용 확장 필드
                    "raw_item_code": item.raw_item_code,
                    "item_code_candidates": item.item_code_candidates,
                    "description_tokens": {
                        "type": item.description_tokens.type if item.description_tokens else None,
                        "length": item.description_tokens.length if item.description_tokens else None,
                        "pcs": item.description_tokens.pcs if item.description_tokens else None,
                        "style": item.description_tokens.style if item.description_tokens else None,
                    } if item.description_tokens else None
                }
                for item in all_line_items
            ],
            "line_item_count": len(all_line_items),
            "page_results": page_results
        }
    except Exception as e:
        logger.error(f"SNG Multi-page OCR Error: {str(e)}")
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


def process_sng_ocr_job_sync(job_id: str, temp_path: str):
    """
    별도 스레드에서 SNG OCR 처리 (동기 함수)

    HTTP 응답과 완전히 분리되어 실행됨
    """
    try:
        logger.info(f"Job {job_id}: OCR 처리 시작")
        jobs[job_id]["status"] = "processing"

        # 1. 임시 파일에서 이미지 데이터 읽기
        with open(temp_path, "rb") as f:
            image_data = f.read()
        logger.info(f"Job {job_id}: 이미지 로드 완료, size={len(image_data)} bytes")

        # 2. OCR 처리
        raw_text = process_image(image_data)
        logger.info(f"Job {job_id}: OCR 완료, text_length={len(raw_text)}")

        # 3. SNG 파서로 구조화된 데이터 추출
        result = parse_sng_invoice(raw_text)

        # 4. 라인 아이템 변환
        line_items = [
            {
                "item_code": item.item_code,
                "color": item.color,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "description": item.description,
                "raw_item_code": item.raw_item_code,
                "item_code_candidates": item.item_code_candidates,
                "description_tokens": {
                    "type": item.description_tokens.type if item.description_tokens else None,
                    "length": item.description_tokens.length if item.description_tokens else None,
                    "pcs": item.description_tokens.pcs if item.description_tokens else None,
                    "style": item.description_tokens.style if item.description_tokens else None,
                } if item.description_tokens else None
            }
            for item in result.line_items
        ]

        # 5. 결과 저장
        jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "company": "SNG",
            "header": {
                "invoice_number": result.header.invoice_number,
                "invoice_date": result.header.invoice_date
            },
            "line_items": line_items,
            "line_item_count": len(line_items),
            "raw_text": raw_text[:1000] if raw_text else ""
        })

        logger.info(f"Job {job_id}: 완료, {len(line_items)} items 추출")

    except Exception as e:
        logger.error(f"Job {job_id}: 실패 - {str(e)}")
        jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e)
        })

    finally:
        # 6. 임시 파일 삭제
        try:
            os.unlink(temp_path)
            logger.info(f"Job {job_id}: 임시 파일 삭제 완료")
        except Exception as e:
            logger.warning(f"Job {job_id}: 임시 파일 삭제 실패 - {str(e)}")


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
