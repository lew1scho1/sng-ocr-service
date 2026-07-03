import pytesseract
from PIL import Image
import io
import re
from typing import List


def process_image(image_data: bytes) -> str:
    """
    이미지 데이터를 받아 OCR 처리 후 텍스트 반환
    """
    # 바이트 데이터를 PIL Image로 변환
    image = Image.open(io.BytesIO(image_data))

    # 이미지 전처리 (선택적)
    image = preprocess_image(image)

    # Tesseract OCR 실행
    text = pytesseract.image_to_string(image, lang='eng')

    return text


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    OCR 정확도를 높이기 위한 이미지 전처리
    """
    # RGB로 변환 (RGBA나 다른 모드인 경우)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 그레이스케일 변환
    image = image.convert('L')

    # 이미지가 너무 작으면 확대
    width, height = image.size
    if width < 1000:
        scale = 1000 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
