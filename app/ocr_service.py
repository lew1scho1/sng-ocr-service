import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import io
import re
import logging
from typing import List, Tuple, Optional, Dict

# 데이터 모델 및 순수 함수는 ocr_models에서 import (pytesseract 비의존)
from .ocr_models import (
    OcrConfig,
    OcrLine,
    ColorRegionResult,
    DEFAULT_OCR_CONFIG,
    COLOR_REGION_OCR_CONFIG,
    detect_color_regions_bbox,
    merge_by_replacement,
)

# DB 색상 목록 연동
from .color_validator import get_valid_colors, preload_colors

logger = logging.getLogger(__name__)

# 서비스 시작 시 색상 목록 미리 로드
try:
    preload_colors()
except Exception as e:
    logger.warning(f"색상 목록 preload 실패 (API 미연결): {e}")


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
    logger.debug(f"=== RAW OCR TEXT (process_image) ===\n{text}\n=== END RAW OCR TEXT ===")
    return text


def preprocess_image(image: Image.Image, config: OcrConfig = None) -> Image.Image:
    """
    OCR 정확도를 높이기 위한 이미지 전처리

    처리 단계 (순서 중요):
    1. RGB 변환
    2. 확대 (threshold 전에 수행해야 정보 손실 최소화)
    3. 그레이스케일 변환
    4. 대비 향상
    5. 샤프닝
    6. 이진화 (threshold 완화)

    Args:
        image: PIL Image 객체
        config: OcrConfig 설정 (None이면 DEFAULT_OCR_CONFIG 사용)
    """
    if config is None:
        config = DEFAULT_OCR_CONFIG

    # 1. RGB로 변환 (RGBA나 다른 모드인 경우)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 2. 이미지 확대 (threshold 전에 수행)
    # 작은 글자가 threshold에서 뭉개지는 것을 방지
    width, height = image.size
    target_width = int(2000 * config.scale_factor / 2.0)  # 기본 2.0 기준 2000px
    if width < target_width:
        scale = target_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"이미지 확대: {width}x{height} → {new_width}x{new_height}")

    # 3. 그레이스케일 변환
    image = image.convert('L')

    # 4. 대비 향상 (Contrast Enhancement)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(config.contrast)

    # 5. 샤프닝 (텍스트 선명도 향상)
    for _ in range(config.sharpen_passes):
        image = image.filter(ImageFilter.SHARPEN)

    # 6. 이진화
    threshold = config.threshold
    image = image.point(lambda x: 255 if x > threshold else 0, mode='1')
    image = image.convert('L')

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


# ========================================
# PHASE 5+: bbox 기반 색상-수량 영역 분리 OCR
# ========================================

def process_image_with_color_regions(image_data: bytes) -> Tuple[str, List[ColorRegionResult]]:
    """
    PHASE 5+: bbox 기반 색상-수량 영역 분리 OCR

    기존 문제점:
    - 줄 인덱스 → 픽셀 좌표 휴리스틱 (line_height = image_height / total_lines)
    - 재OCR 결과를 append하여 파서가 중복 처리

    개선:
    1. image_to_data()로 bbox 좌표 획득
    2. 색상-수량 패턴 라인의 실제 Y 좌표 사용
    3. 해당 영역 재OCR 후 원본 라인 교체 (block merge)

    Returns:
        (병합된_OCR_텍스트, [색상영역_OCR_결과들])
    """
    # 바이트 데이터를 PIL Image로 변환
    image = Image.open(io.BytesIO(image_data))
    original_image = image.copy()  # 원본 보존 (crop용)
    logger.info(f"원본 이미지: {image.size}, mode={image.mode}")

    # 1. 전처리
    preprocessed = preprocess_image(image)

    # 2. bbox 기반 OCR (image_to_data)
    ocr_lines, scale_factor = ocr_with_bbox(preprocessed, original_image.size)
    logger.info(f"bbox OCR: {len(ocr_lines)} lines, scale_factor={scale_factor:.2f}")

    # RAW OCR 텍스트 로깅 (파싱/처리 전)
    raw_ocr_text = "\n".join([line.text for line in ocr_lines])
    logger.debug(f"=== RAW OCR TEXT (bbox) ===\n{raw_ocr_text}\n=== END RAW OCR TEXT ===")

    if not ocr_lines:
        return "", []

    # 3. 색상-수량 영역 감지 (bbox 좌표 기반 + DB 색상 목록 검증)
    valid_colors = get_valid_colors()
    logger.info(f"DB 색상 목록: {len(valid_colors)}개 로드됨")
    color_regions = detect_color_regions_bbox(ocr_lines, original_image.size, valid_colors)

    if not color_regions:
        # 색상 영역 없으면 원본 텍스트 그대로 반환
        full_text = "\n".join([line.text for line in ocr_lines])
        logger.info("색상-수량 영역 감지되지 않음")
        return full_text, []

    # 4. 각 영역에 대해 고해상도 재OCR
    color_results = []
    for region in color_regions:
        y_start, y_end, line_indices = region
        region_result = ocr_color_region(original_image, y_start, y_end)
        if region_result:
            region_result.original_lines = line_indices
            color_results.append(region_result)
            logger.info(f"색상 영역 OCR: y={y_start}-{y_end}, lines={line_indices}, len={len(region_result.region_text)}")

    # 5. Block Merge: 원본 라인을 재OCR 결과로 교체
    merged_text = merge_by_replacement(ocr_lines, color_results)

    return merged_text, color_results


def ocr_with_bbox(image: Image.Image, original_size: Tuple[int, int]) -> Tuple[List[OcrLine], float]:
    """
    image_to_data()를 사용하여 bbox 좌표와 함께 OCR 수행

    Returns:
        (OcrLine 리스트, scale_factor)
    """
    custom_config = r'--oem 3 --psm 6'

    # image_to_data로 bbox 정보 획득
    data = pytesseract.image_to_data(image, lang='eng', config=custom_config, output_type=pytesseract.Output.DICT)

    # scale factor 계산 (전처리에서 확대된 비율)
    preprocessed_width = image.size[0]
    original_width = original_size[0]
    scale_factor = preprocessed_width / original_width if original_width > 0 else 1.0

    # 라인별로 그룹화
    lines_dict: Dict[int, List[dict]] = {}

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if not text:
            continue

        # block_num, par_num, line_num으로 라인 식별
        line_key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])

        if line_key not in lines_dict:
            lines_dict[line_key] = []

        lines_dict[line_key].append({
            'text': text,
            'top': data['top'][i],
            'height': data['height'][i],
            'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
        })

    # OcrLine 리스트 생성
    ocr_lines = []
    for line_key in sorted(lines_dict.keys()):
        words = lines_dict[line_key]
        if not words:
            continue

        # 라인 텍스트 조합
        line_text = " ".join([w['text'] for w in words])

        # Y 좌표 (원본 이미지 기준으로 변환)
        y_top = min(w['top'] for w in words)
        y_bottom = max(w['top'] + w['height'] for w in words)
        y_top_original = int(y_top / scale_factor)
        y_bottom_original = int(y_bottom / scale_factor)

        # 평균 confidence
        avg_conf = sum(w['conf'] for w in words) / len(words) if words else 0.0

        ocr_lines.append(OcrLine(
            text=line_text,
            y_top=y_top_original,
            y_bottom=y_bottom_original,
            confidence=avg_conf
        ))

    # Y 좌표순 정렬
    ocr_lines.sort(key=lambda x: x.y_top)

    return ocr_lines, scale_factor


def ocr_color_region(
    image: Image.Image,
    y_start: int,
    y_end: int,
    config: OcrConfig = None
) -> Optional[ColorRegionResult]:
    """
    색상-수량 영역만 crop하여 고해상도 OCR 수행

    COLOR_REGION_OCR_CONFIG 사용:
    - 3배 확대 (작은 글자 대응)
    - 강화된 대비 (2.5)
    - 2회 샤프닝
    - 색상 코드 전용 whitelist
    """
    if config is None:
        config = COLOR_REGION_OCR_CONFIG

    width, height = image.size

    # 영역 crop (전체 가로 폭 사용)
    region = image.crop((0, y_start, width, y_end))

    # RGB 변환
    if region.mode != 'RGB':
        region = region.convert('RGB')

    # 확대 (config 기반)
    scale = config.scale_factor
    new_width = int(width * scale)
    new_height = int((y_end - y_start) * scale)
    region = region.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 그레이스케일
    region = region.convert('L')

    # 대비 강화 (config 기반)
    enhancer = ImageEnhance.Contrast(region)
    region = enhancer.enhance(config.contrast)

    # 샤프닝 (config 기반)
    for _ in range(config.sharpen_passes):
        region = region.filter(ImageFilter.SHARPEN)

    # 이진화 (config 기반)
    region = region.point(lambda x: 255 if x > config.threshold else 0, mode='1')
    region = region.convert('L')

    # OCR 실행 (config 기반)
    tesseract_config = f'--oem 3 --psm {config.psm}'
    if config.char_whitelist:
        tesseract_config += f' -c tessedit_char_whitelist={config.char_whitelist}'

    text = pytesseract.image_to_string(region, lang='eng', config=tesseract_config)

    if not text.strip():
        return None

    return ColorRegionResult(
        region_text=text,
        y_start=y_start,
        y_end=y_end,
        confidence=0.8  # TODO: Tesseract conf 값 활용
    )
