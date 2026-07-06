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

from .ocr_service import (
    process_image,
    extract_barcodes,
    process_image_with_color_regions,
    process_image_with_blocks,
    process_image_with_blocks_debug,
)
from .ocr_models import OcrMethodResult
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


@app.post("/api/v1/ocr/debug/sng")
async def debug_ocr_sng(file: UploadFile = File(...)):
    """
    SNG OCR 디버그 엔드포인트

    raw OCR 텍스트 및 파싱 과정 전체를 반환합니다.
    - raw_ocr_text: 전처리 직후 OCR 원본 텍스트 (파싱 전)
    - merged_text: 색상 영역 재OCR 후 병합된 텍스트
    - color_regions: 감지된 색상 영역 정보
    - parsed_result: 최종 파싱 결과
    """
    try:
        from .ocr_service import ocr_with_bbox, preprocess_image
        from PIL import Image
        import io

        image_data = await file.read()

        # 1. 이미지 로드 및 전처리
        image = Image.open(io.BytesIO(image_data))
        original_size = image.size
        preprocessed = preprocess_image(image)

        # 2. bbox 기반 OCR (raw 텍스트)
        ocr_lines, scale_factor = ocr_with_bbox(preprocessed, original_size)
        raw_ocr_text = "\n".join([line.text for line in ocr_lines])

        # 3. 색상 영역 처리 및 병합
        merged_text, color_results = process_image_with_color_regions(image_data)

        # 4. 파싱
        result = parse_sng_invoice(merged_text)

        return {
            "success": True,
            "debug": {
                "raw_ocr_text": raw_ocr_text,  # 파싱 전 원본 OCR
                "raw_ocr_lines": len(ocr_lines),
                "merged_text": merged_text,     # 색상 영역 재OCR 후 병합된 텍스트
                "color_regions_detected": len(color_results),
                "color_region_details": [
                    {
                        "y_start": cr.y_start,
                        "y_end": cr.y_end,
                        "region_text": cr.region_text,
                        "original_lines": cr.original_lines
                    }
                    for cr in color_results
                ]
            },
            "parsed_result": {
                "header": {
                    "invoice_number": result.header.invoice_number,
                    "invoice_date": result.header.invoice_date
                },
                "line_items": [
                    {
                        "item_code_raw": item.item_code_raw,
                        "color_raw": item.color_raw,
                        "quantity": item.quantity,
                        "unit_price": item.unit_price,
                        "description_raw": item.description_raw
                    }
                    for item in result.line_items
                ],
                "line_item_count": len(result.line_items)
            }
        }
    except Exception as e:
        logger.error(f"SNG Debug OCR Error: {str(e)}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/api/v1/ocr/debug/sng/blocks")
async def debug_ocr_sng_blocks(file: UploadFile = File(...)):
    """
    SNG OCR 블록 기반 디버그 엔드포인트

    PHASE 6: 점선 기반 아이템 블록 분리 및 영역별 OCR
    - dotted_lines: 감지된 점선 Y 좌표
    - blocks: 각 아이템 블록의 헤더/색상행 OCR 결과
    - raw_color_rows: 색상 행 OCR 원문
    - merged_text: 파서 호환 형식으로 병합된 텍스트
    - parsed_result: 최종 파싱 결과
    """
    try:
        image_data = await file.read()

        # 블록 기반 OCR 디버그
        debug_result = process_image_with_blocks_debug(image_data)

        # 파싱 (병합된 텍스트로)
        parsed = None
        if debug_result['merged_text']:
            result = parse_sng_invoice(debug_result['merged_text'])
            parsed = {
                "header": {
                    "invoice_number": result.header.invoice_number,
                    "invoice_date": result.header.invoice_date
                },
                "line_items": [
                    {
                        "item_code_raw": item.item_code_raw,
                        "color_raw": item.color_raw,
                        "quantity": item.quantity,
                        "unit_price": item.unit_price,
                        "description_raw": item.description_raw
                    }
                    for item in result.line_items
                ],
                "line_item_count": len(result.line_items)
            }

        return {
            "success": True,
            "method": "block_based",
            "debug": {
                "dotted_lines": debug_result['dotted_lines'],
                "block_count": len(debug_result['blocks']),
                "blocks": debug_result['blocks'],
                "raw_color_rows": debug_result['raw_color_rows'],
                "merged_text": debug_result['merged_text'],
                "merged_text_lines": len(debug_result['merged_text'].split('\n')) if debug_result['merged_text'] else 0
            },
            "parsed_result": parsed
        }
    except Exception as e:
        logger.error(f"SNG Block Debug OCR Error: {str(e)}")
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/api/v1/ocr/jobs/sync/sng")
async def create_ocr_job_sync_sng(file: UploadFile = File(...)):
    """
    SNG (Shake-N-Go) 인보이스 전용 OCR (단일 이미지)

    PHASE 6: 점선 기반 아이템 블록 분리 OCR 적용
    - 점선으로 아이템 블록 경계 감지
    - 블록별 영역 분리 (헤더/색상행) 및 개별 OCR
    - 점선 감지 실패 시 기존 color_regions 방식으로 자동 fallback

    추출 정보:
    - 헤더: Invoice NO, Invoice Date
    - 라인 아이템: ITEM_CODE + COLOR + QUANTITY + UNIT_PRICE
    """
    try:
        image_data = await file.read()

        # OCR 처리 (PHASE 6: 블록 기반 영역 분리 OCR)
        # process_image_with_blocks는 점선 감지 실패 시 자동으로 color_regions 방식으로 fallback
        raw_text, item_blocks, ocr_method = process_image_with_blocks(image_data)

        # OCR 메서드 로그 (명확한 SUCCESS/FALLBACK 표시)
        if ocr_method.method == "block_ocr":
            logger.info(
                f"[SNG-SYNC] OCR_METHOD=block_ocr | "
                f"blocks={ocr_method.blocks_created} | "
                f"dotted_lines={ocr_method.dotted_lines_detected} | "
                f"retries={ocr_method.color_row_retries}"
            )
        else:
            logger.warning(
                f"[SNG-SYNC] OCR_METHOD=color_regions (FALLBACK) | "
                f"reason={ocr_method.fallback_reason} | "
                f"dotted_lines={ocr_method.dotted_lines_detected}"
            )

        # 디버그: 전체 OCR 텍스트 로그 출력
        logger.info("=" * 80)
        logger.info("SNG OCR - FULL RAW TEXT START")
        logger.info("=" * 80)
        logger.info(raw_text)
        logger.info("=" * 80)
        logger.info(f"SNG OCR - FULL RAW TEXT END (Total: {len(raw_text)} chars)")
        logger.info("=" * 80)

        # SNG 파서로 구조화된 데이터 추출 (raw 데이터만)
        result = parse_sng_invoice(raw_text)

        return {
            "success": True,
            "company": "SNG",
            "ocr_method": ocr_method.method,
            "ocr_metadata": {
                "dotted_lines": ocr_method.dotted_lines_detected,
                "blocks": ocr_method.blocks_created,
                "fallback_reason": ocr_method.fallback_reason,
                "retries": ocr_method.color_row_retries
            },
            "header": {
                "invoice_number": result.header.invoice_number,
                "invoice_date": result.header.invoice_date
            },
            "line_items": [
                {
                    "item_code_raw": item.item_code_raw,
                    "color_raw": item.color_raw,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "description_raw": item.description_raw,
                    "qty_ordered": item.qty_ordered,
                    "qty_shipped": item.qty_shipped
                }
                for item in result.line_items
            ],
            "line_item_count": len(result.line_items),
            "raw_text_preview": raw_text[:500] if raw_text else "",
            "last_item_code_raw": result.last_item_code_raw,
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

    PHASE 6: 점선 기반 아이템 블록 분리 OCR 적용

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

            # OCR 처리 (PHASE 6: 블록 기반 영역 분리 OCR)
            raw_text, item_blocks, ocr_method = process_image_with_blocks(image_data)

            # OCR 메서드 로그
            if ocr_method.method == "block_ocr":
                logger.info(
                    f"[SNG-MULTI] Page {i+1}/{len(files)} | OCR_METHOD=block_ocr | "
                    f"blocks={ocr_method.blocks_created}"
                )
            else:
                logger.warning(
                    f"[SNG-MULTI] Page {i+1}/{len(files)} | OCR_METHOD=color_regions (FALLBACK) | "
                    f"reason={ocr_method.fallback_reason}"
                )

            # SNG 파서로 구조화된 데이터 추출 (이전 페이지 정보 전달)
            result = parse_sng_invoice(raw_text, prev_item_code, prev_unit_price)

            # 첫 페이지에서 헤더 추출
            if header is None and result.header.invoice_number:
                header = result.header

            # 라인 아이템 병합
            all_line_items.extend(result.line_items)

            # 다음 페이지를 위해 마지막 아이템 정보 저장
            prev_item_code = result.last_item_code_raw
            prev_unit_price = result.last_unit_price

            # 페이지별 결과 저장
            page_results.append({
                "page": i + 1,
                "filename": file.filename,
                "item_count": len(result.line_items),
                "last_item_code_raw": result.last_item_code_raw,
                "ocr_method": ocr_method.method,
                "blocks_created": ocr_method.blocks_created
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
                    "item_code_raw": item.item_code_raw,
                    "color_raw": item.color_raw,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "description_raw": item.description_raw,
                    "qty_ordered": item.qty_ordered,
                    "qty_shipped": item.qty_shipped
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

    PHASE 6: 점선 기반 아이템 블록 분리 OCR 적용
    """
    try:
        logger.info(f"Job {job_id}: OCR 처리 시작")
        jobs[job_id]["status"] = "processing"

        # 1. 임시 파일에서 이미지 데이터 읽기
        with open(temp_path, "rb") as f:
            image_data = f.read()
        logger.info(f"Job {job_id}: 이미지 로드 완료, size={len(image_data)} bytes")

        # 2. OCR 처리 (PHASE 6: 블록 기반 영역 분리 OCR)
        raw_text, item_blocks, ocr_method = process_image_with_blocks(image_data)

        # OCR 메서드 로그
        if ocr_method.method == "block_ocr":
            logger.info(
                f"[SNG-ASYNC] Job {job_id} | OCR_METHOD=block_ocr | "
                f"blocks={ocr_method.blocks_created}"
            )
        else:
            logger.warning(
                f"[SNG-ASYNC] Job {job_id} | OCR_METHOD=color_regions (FALLBACK) | "
                f"reason={ocr_method.fallback_reason}"
            )

        # 3. SNG 파서로 구조화된 데이터 추출
        result = parse_sng_invoice(raw_text)

        # 4. 라인 아이템 변환 (raw 데이터만 - 동기 API와 동일 스키마)
        line_items = [
            {
                "item_code_raw": item.item_code_raw,
                "color_raw": item.color_raw,  # 동기 API와 동일하게 color_raw 사용
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "description_raw": item.description_raw,
                "qty_ordered": item.qty_ordered,
                "qty_shipped": item.qty_shipped
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
