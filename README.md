# SNG OCR Service

인보이스 이미지에서 바코드를 추출하는 OCR 마이크로서비스.

## API Endpoints

### 동기 처리 (권장)
```
POST /api/v1/ocr/jobs/sync
Content-Type: multipart/form-data

file: <image_file>
```

응답:
```json
{
  "success": true,
  "barcodes": ["123456789012", "123456789013"],
  "barcode_count": 2,
  "raw_text_preview": "..."
}
```

### 비동기 처리
```
POST /api/v1/ocr/jobs
Content-Type: multipart/form-data

file: <image_file>
```

```
GET /api/v1/ocr/jobs/{job_id}
```

## 로컬 실행

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Tesseract 설치 (Windows)
# https://github.com/UB-Mannheim/tesseract/wiki

# 서버 실행
uvicorn app.main:app --reload
```

## Docker 실행

```bash
docker build -t sng-ocr-service .
docker run -p 8000:8000 sng-ocr-service
```

## Render 배포

1. GitHub에 푸시
2. Render에서 New > Web Service
3. Connect repository
4. Environment: Docker
5. Deploy
