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
    # Block-based detection
    ItemBlock,
    BlockRegion,
    BlockDetectionConfig,
    DEFAULT_BLOCK_DETECTION_CONFIG,
    detect_dotted_lines,
    create_item_blocks,
    evaluate_color_row_quality,
    COLOR_ROW_OCR_CONFIG,
    COLOR_ROW_RETRY_CONFIG,
    HEADER_OCR_CONFIG,
    COLOR_ROW_RETRY_THRESHOLD,
    OcrMethodResult,
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


# ========================================
# PHASE 6: Block-based OCR (점선 기반)
# ========================================

def get_binary_image_data(image: Image.Image) -> List[List[int]]:
    """
    PIL Image를 2D 픽셀 배열로 변환

    Args:
        image: 그레이스케일 이미지 (L 모드)

    Returns:
        2D 리스트 [y][x] = pixel_value (0-255)
    """
    if image.mode != 'L':
        image = image.convert('L')

    width, height = image.size
    pixels = list(image.getdata())

    # 2D 배열로 변환
    return [pixels[y * width:(y + 1) * width] for y in range(height)]


def process_image_with_blocks(
    image_data: bytes,
    block_config: BlockDetectionConfig = None
) -> Tuple[str, List[ItemBlock], OcrMethodResult]:
    """
    Block-based OCR: 점선 기반 아이템 블록 분리 및 영역별 OCR

    전략:
    1. 이미지에서 점선 감지 → 아이템 블록 경계 결정
    2. 각 블록 내 영역 분리 (header, price, color_row)
    3. color_row 영역에 PSM 11 (sparse text) 적용
    4. 품질 낮으면 PSM 6으로 재시도
    5. 기존 파서 호환 형식으로 병합

    Args:
        image_data: 이미지 바이트 데이터
        block_config: 블록 감지 설정

    Returns:
        (병합된_OCR_텍스트, [ItemBlock들], OcrMethodResult)
    """
    if block_config is None:
        block_config = DEFAULT_BLOCK_DETECTION_CONFIG

    # 이미지 로드
    image = Image.open(io.BytesIO(image_data))
    original_image = image.copy()
    width, height = image.size
    logger.info(f"[BlockOCR] 원본 이미지: {width}x{height}, mode={image.mode}")

    # 1. 전처리 (점선 감지용 - 이진화)
    preprocessed = preprocess_image(image)
    binary_data = get_binary_image_data(preprocessed)

    # 점선 좌표를 원본 이미지 기준으로 변환
    scale_factor = preprocessed.size[0] / width if width > 0 else 1.0

    # 2. 점선 감지
    dotted_ys = detect_dotted_lines(
        binary_data,
        preprocessed.size[0],
        preprocessed.size[1],
        block_config
    )

    # 원본 좌표로 변환
    dotted_ys_original = [int(y / scale_factor) for y in dotted_ys]
    logger.info(f"[BlockOCR] 점선 감지: {len(dotted_ys_original)}개")

    # 점선이 없으면 기존 방식으로 폴백
    if len(dotted_ys_original) < 2:
        logger.warning("[BlockOCR] FALLBACK - 점선 부족 (dotted_lines < 2)")
        text, _ = process_image_with_color_regions(image_data)
        method_result = OcrMethodResult(
            method="color_regions",
            dotted_lines_detected=len(dotted_ys_original),
            blocks_created=0,
            fallback_reason="dotted_lines_insufficient"
        )
        return text, [], method_result

    # 3. 아이템 블록 생성
    blocks = create_item_blocks(
        dotted_ys_original,
        width,
        height,
        block_config
    )

    if not blocks:
        logger.warning("[BlockOCR] FALLBACK - 블록 생성 실패")
        text, _ = process_image_with_color_regions(image_data)
        method_result = OcrMethodResult(
            method="color_regions",
            dotted_lines_detected=len(dotted_ys_original),
            blocks_created=0,
            fallback_reason="no_valid_blocks"
        )
        return text, [], method_result

    # 4. 각 블록의 영역별 OCR (재시도 횟수 추적)
    total_retries = 0
    for block in blocks:
        retried = _ocr_block_regions(original_image, block)
        if retried:
            total_retries += 1

    # 5. 파서 호환 텍스트로 병합
    merged_text = _merge_block_results(blocks)

    # 성공 결과
    method_result = OcrMethodResult(
        method="block_ocr",
        dotted_lines_detected=len(dotted_ys_original),
        blocks_created=len(blocks),
        fallback_reason=None,
        color_row_retries=total_retries
    )

    logger.info(
        f"[BlockOCR] SUCCESS - blocks={len(blocks)}, "
        f"dotted_lines={len(dotted_ys_original)}, retries={total_retries}"
    )

    return merged_text, blocks, method_result


def _ocr_block_regions(image: Image.Image, block: ItemBlock) -> bool:
    """
    블록 내 각 영역에 대해 OCR 수행 (in-place)

    Args:
        image: 원본 이미지
        block: 처리할 ItemBlock

    Returns:
        bool: 색상 행 재시도 여부
    """
    retried = False

    # Header 영역 OCR
    if block.header_region:
        header_text = _ocr_region(image, block.header_region, HEADER_OCR_CONFIG)
        block.header_text = header_text
        logger.debug(f"[Block {block.index}] Header: {header_text[:50]}...")

    # Color Row 영역 OCR (핵심)
    if block.color_row_region:
        color_text = _ocr_region(image, block.color_row_region, COLOR_ROW_OCR_CONFIG)

        # 품질 평가
        quality, pair_count = evaluate_color_row_quality(color_text)
        block.color_row_confidence = quality

        logger.debug(
            f"[Block {block.index}] ColorRow 1차: quality={quality:.2f}, "
            f"pairs={pair_count}, text='{color_text[:60]}...'"
        )

        # 품질 낮으면 재시도
        if quality < COLOR_ROW_RETRY_THRESHOLD:
            logger.info(f"[Block {block.index}] ColorRow 재시도 (quality={quality:.2f})")
            retry_text = _ocr_region(image, block.color_row_region, COLOR_ROW_RETRY_CONFIG)
            retry_quality, retry_pairs = evaluate_color_row_quality(retry_text)
            retried = True

            if retry_quality > quality:
                color_text = retry_text
                block.color_row_confidence = retry_quality
                logger.debug(
                    f"[Block {block.index}] ColorRow 재시도 성공: "
                    f"quality={retry_quality:.2f}, pairs={retry_pairs}"
                )

        block.color_row_text = color_text

    return retried


def _ocr_region(
    image: Image.Image,
    region: BlockRegion,
    config: OcrConfig
) -> str:
    """
    특정 영역에 대해 OCR 수행

    Args:
        image: 원본 이미지
        region: 크롭할 영역
        config: OCR 설정

    Returns:
        OCR 텍스트
    """
    width, height = image.size

    # 영역 좌표 (전체 폭이면 0, image_width)
    x_start = region.x_start if region.x_start > 0 else 0
    x_end = region.x_end if region.x_end > 0 else width

    # 안전한 좌표
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    y_start = max(0, region.y_start)
    y_end = min(height, region.y_end)

    if x_end <= x_start or y_end <= y_start:
        return ""

    # 영역 크롭
    cropped = image.crop((x_start, y_start, x_end, y_end))

    # RGB 변환
    if cropped.mode != 'RGB':
        cropped = cropped.convert('RGB')

    # 확대
    crop_width, crop_height = cropped.size
    scale = config.scale_factor
    new_width = int(crop_width * scale)
    new_height = int(crop_height * scale)
    cropped = cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 그레이스케일
    cropped = cropped.convert('L')

    # 대비 강화
    enhancer = ImageEnhance.Contrast(cropped)
    cropped = enhancer.enhance(config.contrast)

    # 샤프닝
    for _ in range(config.sharpen_passes):
        cropped = cropped.filter(ImageFilter.SHARPEN)

    # 이진화
    cropped = cropped.point(lambda x: 255 if x > config.threshold else 0, mode='1')
    cropped = cropped.convert('L')

    # OCR 실행
    tesseract_config = f'--oem 3 --psm {config.psm}'
    if config.char_whitelist:
        tesseract_config += f' -c tessedit_char_whitelist={config.char_whitelist}'

    text = pytesseract.image_to_string(cropped, lang='eng', config=tesseract_config)

    return text.strip()


def _merge_block_results(blocks: List[ItemBlock]) -> str:
    """
    블록 OCR 결과를 파서 호환 텍스트로 병합

    파서 기대 형식:
    - 아이템 헤더 라인
    - 가격 라인
    - 색상-수량 라인들

    Args:
        blocks: OCR 완료된 ItemBlock 리스트

    Returns:
        병합된 텍스트
    """
    lines = []

    for block in blocks:
        # Header (아이템 코드, 설명 포함)
        if block.header_text:
            # 헤더 내 각 줄 추가
            for line in block.header_text.split('\n'):
                cleaned = line.strip()
                if cleaned:
                    lines.append(cleaned)

        # Color Row (색상-수량 데이터)
        if block.color_row_text:
            for line in block.color_row_text.split('\n'):
                cleaned = line.strip()
                if cleaned:
                    lines.append(cleaned)

        logger.debug(
            f"[Merge] Block[{block.index}]: header={len(block.header_text)}chars, "
            f"color={len(block.color_row_text)}chars"
        )

    result = '\n'.join(lines)
    logger.info(f"[BlockMerge] 총 {len(lines)}줄 병합")

    return result


def process_image_with_blocks_debug(
    image_data: bytes,
    block_config: BlockDetectionConfig = None
) -> Dict:
    """
    Block-based OCR 디버그 버전 - 상세 정보 반환

    Returns:
        {
            'merged_text': str,
            'blocks': [ItemBlock...],
            'dotted_lines': [int...],
            'raw_color_rows': [str...]
        }
    """
    if block_config is None:
        block_config = DEFAULT_BLOCK_DETECTION_CONFIG

    image = Image.open(io.BytesIO(image_data))
    original_image = image.copy()
    width, height = image.size

    # 전처리
    preprocessed = preprocess_image(image)
    binary_data = get_binary_image_data(preprocessed)

    scale_factor = preprocessed.size[0] / width if width > 0 else 1.0

    # 점선 감지
    dotted_ys = detect_dotted_lines(
        binary_data,
        preprocessed.size[0],
        preprocessed.size[1],
        block_config
    )
    dotted_ys_original = [int(y / scale_factor) for y in dotted_ys]

    # 블록 생성
    blocks = []
    if len(dotted_ys_original) >= 2:
        blocks = create_item_blocks(
            dotted_ys_original,
            width,
            height,
            block_config
        )

        for block in blocks:
            _ocr_block_regions(original_image, block)

    merged_text = _merge_block_results(blocks) if blocks else ""

    return {
        'merged_text': merged_text,
        'blocks': [
            {
                'index': b.index,
                'y_start': b.y_start,
                'y_end': b.y_end,
                'height': b.height,
                'header_text': b.header_text,
                'color_row_text': b.color_row_text,
                'color_row_confidence': b.color_row_confidence,
            }
            for b in blocks
        ],
        'dotted_lines': dotted_ys_original,
        'raw_color_rows': [b.color_row_text for b in blocks if b.color_row_text]
    }
