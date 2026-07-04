"""
OCR 데이터 모델 및 순수 함수 (pytesseract 비의존)

테스트 시 pytesseract 없이 import 가능
"""
import re
import logging
from typing import List, Tuple, Optional, Set, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# OCR 설정
# =============================================================================

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


# =============================================================================
# 데이터 모델
# =============================================================================

@dataclass
class OcrLine:
    """bbox 기반 OCR 라인"""
    text: str                  # 라인 텍스트
    y_top: int                 # 라인 상단 Y 좌표
    y_bottom: int              # 라인 하단 Y 좌표
    confidence: float          # OCR 신뢰도
    is_color_region: bool = False  # 색상-수량 영역 여부


class ColorPatternType(Enum):
    """색상-수량 패턴 유형"""
    SIMPLE = "simple"        # 1B - 2, 30 - 6
    SLASH = "slash"          # P4/30 - 2, P27/30 - 6
    COMPOUND = "compound"    # C-42730 - 4
    NUMERIC = "numeric"      # 2 - 5, 613 - 3


@dataclass
class ColorLineCandidate:
    """색상 라인 후보 (점수 기반)"""
    line_index: int
    text: str
    score: float                    # 0.0 ~ 1.0
    matched_patterns: List[str]     # 매칭된 패턴 유형들
    color_qty_pairs: List[Tuple[str, int]]  # [(색상, 수량), ...]
    pattern_types: List[ColorPatternType]
    db_match_count: int = 0         # DB 매칭 색상 수 (가산점용)


@dataclass
class ColorRegionCandidate:
    """색상 영역 후보 (그룹)"""
    y_start: int
    y_end: int
    line_indices: List[int]
    total_score: float
    candidate_lines: List[ColorLineCandidate]
    raw_line_texts: List[str]


@dataclass
class ColorRegionResult:
    """색상-수량 영역 OCR 결과"""
    region_text: str           # 해당 영역의 OCR 텍스트
    y_start: int               # 영역 시작 Y 좌표
    y_end: int                 # 영역 끝 Y 좌표
    confidence: float          # 감지 신뢰도
    original_lines: List[int] = field(default_factory=list)  # 대체될 원본 라인 인덱스


# =============================================================================
# 색상 영역 감지 (Score-based Candidate Collector)
# =============================================================================

# 색상-수량 패턴 정의 (유형별)
COLOR_QTY_PATTERNS = {
    # Simple: 1B - 2, 30 - 6, 2 - 5
    ColorPatternType.SIMPLE: [
        r'\b([A-Z]?\d{1,3}[A-Z]?)\s*[-–—]\s*(\d{1,3})\b',  # 1B-2, 30-6
        r'\b([A-Z]{1,2}\d?)\s*[-–—]\s*(\d{1,3})\b',        # 1B-2, P4-3
    ],
    # Slash: P4/30 - 2, P27/30 - 6
    ColorPatternType.SLASH: [
        r'\b([A-Z]?\d{1,3}/\d{1,3})\s*[-–—]\s*(\d{1,3})\b',    # P4/30-2
        r'\b([A-Z]{1,2}\d{1,2}/\d{1,3})\s*[-–—]\s*(\d{1,3})\b', # P27/30-2
    ],
    # Compound: C-42730 - 4
    ColorPatternType.COMPOUND: [
        r'\b([A-Z]-?\d{4,6})\s*[-–—]\s*(\d{1,3})\b',       # C-42730-4
        r'\b([A-Z]{2,}-\d{2,})\s*[-–—]\s*(\d{1,3})\b',     # OT-27-3
    ],
    # Numeric: 2 - 5, 613 - 3 (순수 숫자)
    ColorPatternType.NUMERIC: [
        r'(?:^|\s)(\d{1,3})\s*[-–—]\s*(\d{1,3})(?:\s|$)',  # 2-5, 613-3
    ],
}

# 제외 패턴 (명확한 비색상 라인)
EXCLUDE_PATTERNS = [
    r'\bINVOICE\s*(NO|NUMBER|#|DATE)',  # Invoice 헤더
    r'\b(PACKED|ORDERED|SHIPPED)\b.*\b(DESCRIPTION|PRICE|QTY)',  # 테이블 헤더
    r'\bITEM\s*(NUMBER|CODE|#)',  # 아이템 헤더
    r'\d{10,}',  # 10자리 이상 연속 숫자 (바코드, 인보이스 번호)
    r'\bPAGE\s+\d+\s*(OF|/)',  # 페이지 번호
    r'\b(SUB\s*)?TOTAL\s*:?\s*\$',  # 합계 금액
]

# 감점 패턴 (색상 줄 가능성 낮음)
PENALTY_PATTERNS = [
    (r'\b(ORGANIQUE|WATER|BODY|WAVE|CURL|TWIST|BRAID|DEEP|LOOSE|NATURAL)\b', -0.3),  # 설명 키워드
    (r'\bS[A-Z]{4,}\d*\b', -0.4),  # 아이템 코드 (SOATX, SOBDX24)
    (r'\b\d{1,3}\.\d{2}\s*\d{1,3}\.\d{2}\b', -0.2),  # 연속 가격 (7.00 42.00)
]

# 가산점 신호
BONUS_SIGNALS = [
    (r'[-–—]\s*\d{1,2}\s+[-–—]\s*\d{1,2}', 0.3),  # 복수 color-qty 토큰 (1B-2 30-6)
    (r'^[A-Z0-9/\-\s]+$', 0.2),  # 라인이 색상 문자만 포함
]

# 점수 임계값
CANDIDATE_SCORE_THRESHOLD = 0.3


def detect_color_regions_bbox(
    ocr_lines: List[OcrLine],
    image_size: Tuple[int, int],
    valid_colors: Optional[Set[str]] = None
) -> List[Tuple[int, int, List[int]]]:
    """
    색상-수량 영역 후보 감지 (Score-based Candidate Collector)

    전략:
    - 감지 단계: 넓은 recall, 구조 점수 기반 후보 수집
    - 검증 단계: 파서/DB에서 precision 회수 (후단 처리)

    Args:
        ocr_lines: OCR 라인 목록
        image_size: 이미지 크기 (width, height)
        valid_colors: DB 색상 목록 (가산점용, 탈락 조건 아님)

    Returns:
        [(y_start, y_end, [line_indices]), ...] - 후보 영역
    """
    if not ocr_lines:
        return []

    # Phase 1: 라인별 점수 계산
    line_candidates = []
    for i, line in enumerate(ocr_lines):
        candidate = score_color_line(i, line.text, valid_colors)
        if candidate and candidate.score >= CANDIDATE_SCORE_THRESHOLD:
            line.is_color_region = True
            line_candidates.append(candidate)
            logger.debug(
                f"색상 후보: line[{i}] score={candidate.score:.2f} "
                f"patterns={candidate.matched_patterns} text='{line.text[:50]}'"
            )

    if not line_candidates:
        logger.info("색상 후보 라인 없음")
        return []

    logger.info(f"색상 후보 라인: {len(line_candidates)}개 (threshold={CANDIDATE_SCORE_THRESHOLD})")

    # Phase 2: 인접 라인 그룹화
    regions = group_candidate_lines(line_candidates, ocr_lines, image_size)

    logger.info(f"색상 영역 감지: {len(regions)}개 영역")
    for region in regions:
        logger.debug(
            f"  영역: y={region[0]}-{region[1]}, lines={region[2]}"
        )

    return regions


def score_color_line(
    line_index: int,
    text: str,
    valid_colors: Optional[Set[str]] = None
) -> Optional[ColorLineCandidate]:
    """
    라인별 색상 후보 점수 계산

    점수 구성:
    - 기본: 패턴 매칭 (유형별 가중치)
    - 가산: 복수 토큰, 짧은 라인, DB 매칭
    - 감점: 설명 키워드, 아이템 코드 패턴
    """
    text_upper = text.upper().strip()

    if not text_upper:
        return None

    # 제외 패턴 확인 (즉시 탈락)
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text_upper):
            return None

    # 패턴 매칭 및 점수 계산
    score = 0.0
    matched_patterns = []
    all_color_qty_pairs = []
    pattern_types = []

    # 유형별 패턴 매칭
    for pattern_type, patterns in COLOR_QTY_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                # 패턴 유형별 기본 점수
                type_scores = {
                    ColorPatternType.SIMPLE: 0.5,
                    ColorPatternType.SLASH: 0.6,
                    ColorPatternType.COMPOUND: 0.5,
                    ColorPatternType.NUMERIC: 0.3,
                }
                base_score = type_scores.get(pattern_type, 0.4)

                for match in matches:
                    color, qty = match[0], int(match[1])
                    # 길이 제한 (1-10자)
                    if 1 <= len(color) <= 10:
                        score += base_score
                        matched_patterns.append(f"{pattern_type.value}:{color}")
                        all_color_qty_pairs.append((color, qty))
                        if pattern_type not in pattern_types:
                            pattern_types.append(pattern_type)

    if not all_color_qty_pairs:
        return None

    # 복수 토큰 보너스
    if len(all_color_qty_pairs) >= 2:
        score += 0.3
        matched_patterns.append("bonus:multi_token")

    # 라인 길이 보너스 (짧은 라인이 색상 줄일 가능성 높음)
    if len(text_upper) <= 30:
        score += 0.1
    elif len(text_upper) <= 50:
        score += 0.05

    # 가산점 신호
    for pattern, bonus in BONUS_SIGNALS:
        if re.search(pattern, text_upper):
            score += bonus
            matched_patterns.append(f"bonus:{pattern[:20]}")

    # 감점 패턴
    for pattern, penalty in PENALTY_PATTERNS:
        if re.search(pattern, text_upper):
            score += penalty  # penalty는 음수
            matched_patterns.append(f"penalty:{pattern[:20]}")

    # DB 색상 매칭 (가산점, 탈락 조건 아님)
    db_match_count = 0
    if valid_colors:
        for color, _ in all_color_qty_pairs:
            if is_color_in_db_fuzzy(color, valid_colors):
                db_match_count += 1
                score += 0.2  # DB 매칭 가산점
                matched_patterns.append(f"db_match:{color}")

    # 점수 정규화 (0-1 범위)
    score = max(0.0, min(1.0, score))

    return ColorLineCandidate(
        line_index=line_index,
        text=text_upper,
        score=score,
        matched_patterns=matched_patterns,
        color_qty_pairs=all_color_qty_pairs,
        pattern_types=pattern_types,
        db_match_count=db_match_count
    )


def group_candidate_lines(
    candidates: List[ColorLineCandidate],
    ocr_lines: List[OcrLine],
    image_size: Tuple[int, int]
) -> List[Tuple[int, int, List[int]]]:
    """
    후보 라인 그룹화 (인접성 + 컨텍스트)

    개선:
    - Y 거리 허용폭 확대 (70px)
    - 중간 라인 포함 (후보 사이 비후보 라인)
    - 너무 큰 그룹 분할
    """
    if not candidates:
        return []

    # 라인 인덱스 순 정렬
    sorted_candidates = sorted(candidates, key=lambda c: c.line_index)

    groups = []
    current_group = [sorted_candidates[0]]

    for candidate in sorted_candidates[1:]:
        prev_candidate = current_group[-1]
        prev_line = ocr_lines[prev_candidate.line_index]
        curr_line = ocr_lines[candidate.line_index]

        # Y 거리 계산
        y_gap = curr_line.y_top - prev_line.y_bottom

        # 라인 인덱스 거리 (중간 라인 수)
        line_gap = candidate.line_index - prev_candidate.line_index - 1

        # 그룹화 조건:
        # 1. Y 거리 70px 이내
        # 2. 중간 라인 2개 이하
        if y_gap <= 70 and line_gap <= 2:
            current_group.append(candidate)
        else:
            groups.append(current_group)
            current_group = [candidate]

    groups.append(current_group)

    # 영역 좌표 계산
    regions = []
    image_height = image_size[1]

    for group in groups:
        line_indices = [c.line_index for c in group]

        # 그룹 내 모든 라인의 Y 범위
        y_start = min(ocr_lines[i].y_top for i in line_indices)
        y_end = max(ocr_lines[i].y_bottom for i in line_indices)

        # 중간 라인 포함 (후보 사이 비후보 라인)
        min_idx = min(line_indices)
        max_idx = max(line_indices)
        expanded_indices = list(range(min_idx, max_idx + 1))

        # 마진 추가 (위아래 25px)
        y_start = max(0, y_start - 25)
        y_end = min(image_height, y_end + 25)

        # 최소 크기 확인
        if y_end - y_start >= 25:
            regions.append((y_start, y_end, expanded_indices))

    return regions


# =============================================================================
# DB 색상 매칭 (Fuzzy)
# =============================================================================

def is_color_in_db_fuzzy(color: str, valid_colors: Set[str]) -> bool:
    """
    색상이 DB 목록에 있는지 퍼지 매칭

    Args:
        color: 추출된 색상 코드
        valid_colors: 유효한 색상 목록 (대문자)

    Returns:
        매칭되면 True
    """
    if not color or not valid_colors:
        return False

    color_upper = color.upper().strip()

    # 정확히 일치
    if color_upper in valid_colors:
        return True

    # OCR 오인식 변형 확인
    variants = generate_color_variants(color_upper)
    for variant in variants:
        if variant in valid_colors:
            return True

    return False


def generate_color_variants(color: str) -> List[str]:
    """
    OCR 오인식을 고려한 색상 변형 생성

    Args:
        color: 색상 코드 (대문자)

    Returns:
        변형 목록
    """
    variants = []

    # I ↔ 1 변환
    if 'I' in color:
        variants.append(color.replace('I', '1'))
    if '1' in color:
        variants.append(color.replace('1', 'I'))

    # O ↔ 0 변환
    if 'O' in color:
        variants.append(color.replace('O', '0'))
    if '0' in color:
        variants.append(color.replace('0', 'O'))

    # 첫 글자 변환 (IB → 1B, OB → 0B 등)
    if len(color) >= 2:
        first_char = color[0]
        rest = color[1:]
        if first_char == 'I':
            variants.append('1' + rest)
        elif first_char == 'O':
            variants.append('0' + rest)
        elif first_char == 'L':
            variants.append('1' + rest)
        elif first_char == 'S':
            variants.append('5' + rest)
        elif first_char == 'Z':
            variants.append('2' + rest)

    return variants


# =============================================================================
# 텍스트 병합
# =============================================================================

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


# =============================================================================
# 하위 호환성 (deprecated)
# =============================================================================

def _is_color_in_db(color: str, valid_colors: Set[str]) -> bool:
    """Deprecated: use is_color_in_db_fuzzy instead"""
    return is_color_in_db_fuzzy(color, valid_colors)


def _generate_color_variants(color: str) -> List[str]:
    """Deprecated: use generate_color_variants instead"""
    return generate_color_variants(color)
