"""
OCR 데이터 모델 및 순수 함수 (pytesseract 비의존)

테스트 시 pytesseract 없이 import 가능
"""
import re
import logging
from typing import List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OcrConfig:
    """OCR 전처리 및 설정 구성"""
    scale_factor: float = 2.0           # 이미지 확대 비율
    contrast: float = 1.8               # 대비 향상 정도
    threshold: int = 150                # 이진화 임계값
    sharpen_passes: int = 1             # 샤프닝 횟수
    psm: int = 6                        # Tesseract PSM 모드
    char_whitelist: str = ""            # 허용 문자 (빈 문자열 = 전체)


# 기본 OCR 설정 (일반 텍스트)
DEFAULT_OCR_CONFIG = OcrConfig(
    scale_factor=2.0,
    contrast=1.8,
    threshold=150,
    sharpen_passes=1,
    psm=6,
    char_whitelist=""
)

# 색상-수량 영역 전용 OCR 설정
COLOR_REGION_OCR_CONFIG = OcrConfig(
    scale_factor=3.0,               # 더 높은 확대 (작은 글자)
    contrast=2.5,                   # 더 강한 대비
    threshold=140,                  # 더 낮은 임계값 (더 많은 픽셀 보존)
    sharpen_passes=2,               # 2회 샤프닝
    psm=6,                          # 블록 모드
    char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/-"  # 색상 코드용
)


@dataclass
class OcrLine:
    """bbox 기반 OCR 라인"""
    text: str                  # 라인 텍스트
    y_top: int                 # 라인 상단 Y 좌표
    y_bottom: int              # 라인 하단 Y 좌표
    confidence: float          # OCR 신뢰도
    is_color_region: bool = False  # 색상-수량 영역 여부


@dataclass
class ColorRegionResult:
    """색상-수량 영역 OCR 결과"""
    region_text: str           # 해당 영역의 OCR 텍스트
    y_start: int               # 영역 시작 Y 좌표
    y_end: int                 # 영역 끝 Y 좌표
    confidence: float          # 감지 신뢰도
    original_lines: List[int] = field(default_factory=list)  # 대체될 원본 라인 인덱스


def detect_color_regions_bbox(
    ocr_lines: List[OcrLine],
    image_size: Tuple[int, int]
) -> List[Tuple[int, int, List[int]]]:
    """
    bbox 좌표 기반으로 색상-수량 영역 감지

    Returns:
        [(y_start, y_end, [line_indices]), ...] - 원본 이미지 픽셀 좌표 및 해당 라인 인덱스
    """
    if not ocr_lines:
        return []

    # 색상-수량 패턴
    color_qty_pattern = r'[A-Z0-9][A-Z0-9/]*\s*[-–—]\s*\d{1,3}'

    # 패턴 매칭되는 라인 찾기
    color_line_indices = []
    for i, line in enumerate(ocr_lines):
        if re.search(color_qty_pattern, line.text.upper()):
            line.is_color_region = True
            color_line_indices.append(i)

    if not color_line_indices:
        return []

    # 연속된 색상 라인 그룹화 (인접 라인 병합)
    groups = []
    current_group = [color_line_indices[0]]

    for idx in color_line_indices[1:]:
        # Y 좌표 기준으로 인접 여부 판단 (50px 이내)
        prev_line = ocr_lines[current_group[-1]]
        curr_line = ocr_lines[idx]

        if curr_line.y_top - prev_line.y_bottom <= 50:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]

    groups.append(current_group)

    # 각 그룹의 Y 좌표 범위 계산
    regions = []
    image_height = image_size[1]

    for group in groups:
        # 그룹 내 모든 라인의 Y 범위
        y_start = min(ocr_lines[i].y_top for i in group)
        y_end = max(ocr_lines[i].y_bottom for i in group)

        # 여유 마진 추가 (위아래 20px)
        y_start = max(0, y_start - 20)
        y_end = min(image_height, y_end + 20)

        # 너무 작은 영역 제외
        if y_end - y_start >= 30:
            regions.append((y_start, y_end, group))

    logger.info(f"bbox 기반 색상 영역 감지: {len(regions)}개 영역")
    return regions


def merge_by_replacement(ocr_lines: List[OcrLine], color_results: List[ColorRegionResult]) -> str:
    """
    Block Merge: 원본 라인을 재OCR 결과로 교체

    기존 append 방식의 문제점:
    - 파서가 중복 컨텍스트를 처리해야 함
    - COLOR_REGION_OCR 마커 파싱 필요

    개선:
    - 재OCR 대상 라인을 찾아서 직접 교체
    - 파서는 깨끗한 단일 텍스트만 처리
    """
    if not color_results:
        return "\n".join([line.text for line in ocr_lines])

    # 교체 대상 라인 인덱스 수집
    replaced_indices = set()
    replacements = {}  # {첫번째_라인_인덱스: 대체_텍스트}

    for result in color_results:
        if result.original_lines:
            # 해당 그룹의 첫 번째 라인 위치에 재OCR 결과 삽입
            first_idx = min(result.original_lines)
            replacements[first_idx] = result.region_text.strip()
            # 나머지 라인은 제거 대상
            for idx in result.original_lines:
                replaced_indices.add(idx)

    # 병합된 텍스트 생성
    merged_lines = []
    for i, line in enumerate(ocr_lines):
        if i in replacements:
            # 재OCR 결과로 교체
            merged_lines.append(replacements[i])
        elif i not in replaced_indices:
            # 교체 대상이 아닌 일반 라인
            merged_lines.append(line.text)
        # replaced_indices에만 있는 경우 (나머지 라인): 건너뜀

    return "\n".join(merged_lines)
