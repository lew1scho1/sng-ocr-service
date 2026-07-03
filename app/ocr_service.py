import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def process_image(image_data: bytes) -> str:
    """
    이미지 데이터를 받아 OCR 처리 후 텍스트 반환
    """
    # 바이트 데이터를 PIL Image로 변환
    image = Image.open(io.BytesIO(image_data))
    logger.info(f"원본 이미지: {image.size}, mode={image.mode}")

    # 이미지 전처리
    image = preprocess_image(image)

    # Tesseract OCR 실행 (PSM 설정 추가)
    # PSM 6: 단일 텍스트 블록으로 가정 (테이블/인보이스에 적합)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, lang='eng', config=custom_config)

    logger.info(f"OCR 결과 길이: {len(text)} chars")
    return text


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    OCR 정확도를 높이기 위한 이미지 전처리

    처리 단계:
    1. RGB 변환
    2. 그레이스케일 변환
    3. 대비 향상
    4. 샤프닝
    5. 이진화 (thresholding)
    6. 필요시 확대
    """
    # 1. RGB로 변환 (RGBA나 다른 모드인 경우)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 2. 그레이스케일 변환
    image = image.convert('L')

    # 3. 대비 향상 (Contrast Enhancement)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # 1.5배 대비 증가

    # 4. 샤프닝 (텍스트 선명도 향상)
    image = image.filter(ImageFilter.SHARPEN)

    # 5. 이진화 (Otsu's method 대신 간단한 threshold)
    # 인보이스는 보통 밝은 배경에 어두운 텍스트
    threshold = 180  # 조정 가능
    image = image.point(lambda x: 255 if x > threshold else 0, mode='1')
    image = image.convert('L')  # 다시 그레이스케일로

    # 6. 이미지가 너무 작으면 확대 (OCR 정확도 향상)
    width, height = image.size
    if width < 1500:
        scale = 1500 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"이미지 확대: {width}x{height} → {new_width}x{new_height}")

    return image


def extract_barcodes(text: str) -> List[str]:
    """
    OCR 텍스트에서 바코드 추출 (12-14자리 숫자)
    UPC-A: 12자리
    EAN-13: 13자리
    GTIN-14: 14자리
    """
    if not text:
        return []

    # 12-14자리 연속 숫자 패턴 추출
    pattern = r'\b\d{12,14}\b'
    matches = re.findall(pattern, text)

    # 중복 제거
    unique_barcodes = list(dict.fromkeys(matches))

    # 유효성 검사: 전부 0인 것 제외
    valid_barcodes = [
        barcode for barcode in unique_barcodes
        if not re.match(r'^0+$', barcode)
    ]

    return valid_barcodes
