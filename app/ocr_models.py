"""
OCR 데이터 모델 및 순수 함수 (pytesseract 비의존)

테스트 시 pytesseract 없이 import 가능
"""
import re
import logging
from typing import List, Tuple, Optional, Set, Dict, Any, Union
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
    is_item_line: bool = False     # 아이템 시작 라인 여부 (블록 문맥용)


class ColorPatternType(Enum):
    """색상-수량 패턴 유형"""
    SIMPLE = "simple"        # 1B - 2, 30 - 6
    SLASH = "slash"          # P4/30 - 2, P27/30 - 6
    COMPOUND = "compound"    # C-42730 - 4, OT-27 - 3
    NUMERIC = "numeric"      # 2 - 5, 613 - 3


@dataclass
class ColorLineCandidate:
    """색상 라인 후보 (점수 기반)"""
    line_index: int
    text: str
    score: float                    # raw score (상대 비교용, 상한 없음)
    matched_patterns: List[str]     # 매칭된 패턴 유형들
    color_qty_pairs: List[Tuple[str, int]]  # [(색상, 수량), ...] - dedupe 적용됨
    pattern_types: List[ColorPatternType]
    db_match_count: int = 0         # DB 매칭 색상 수 (가산점용)
    unique_token_count: int = 0     # 고유 토큰 수


@dataclass
class ColorRegionCandidate:
    """색상 영역 후보 (그룹) - 메타데이터 포함"""
    y_start: int
    y_end: int
    line_indices: List[int]
    total_score: float              # 그룹 내 후보 점수 합계
    avg_score: float                # 평균 점수 (상대 비교용)
    candidate_lines: List[ColorLineCandidate]
    raw_line_texts: List[str]
    context_type: str = "unknown"   # "after_item", "standalone", "multi_item"


@dataclass
class ColorRegionResult:
    """색상-수량 영역 OCR 결과"""
    region_text: str           # 해당 영역의 OCR 텍스트
    y_start: int               # 영역 시작 Y 좌표
    y_end: int                 # 영역 끝 Y 좌표
    confidence: float          # 감지 신뢰도
    original_lines: List[int] = field(default_factory=list)  # 대체될 원본 라인 인덱스


# =============================================================================
# 색상 영역 감지 (Score-based Candidate Collector v2)
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
    # Compound: C-42730 - 4, OT-27 - 3 (OCR 변형 허용 확대)
    ColorPatternType.COMPOUND: [
        r'\b([A-Z][\-\s]?\d{4,6})\s*[-–—]\s*(\d{1,3})\b',   # C-42730-4, C 42730-4
        r'\b([A-Z]{2,}[\-\s]?\d{2,})\s*[-–—]\s*(\d{1,3})\b', # OT-27-3, OT 27-3
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

# 아이템 라인 감지 패턴 (블록 문맥용)
ITEM_LINE_PATTERNS = [
    r'\bS[A-Z]{2,}\d{2,}\b',  # SOATX24, SOBDX24
    r'\b\d+\s+\d+\s+S[A-Z]+',  # 12 12 STEST01
]

# 감점 패턴 (색상 줄 가능성 낮음) - 감점 값 조정
PENALTY_PATTERNS = [
    (r'\b(ORGANIQUE|WATER|BODY|WAVE|CURL|TWIST|BRAID|DEEP|LOOSE|NATURAL)\b', -0.15),
    (r'\bS[A-Z]{4,}\d*\b', -0.25),  # 아이템 코드
    (r'\b\d{1,3}\.\d{2}\s+\d{1,3}\.\d{2}\b', -0.1),  # 연속 가격
]

# 가산점 신호 - 가산 값 조정
BONUS_SIGNALS = [
    (r'[-–—]\s*\d{1,2}\s+.*[-–—]\s*\d{1,2}', 0.15),  # 복수 color-qty 토큰
    (r'^[A-Z0-9/\-\s]+$', 0.1),  # 라인이 색상 문자만 포함
]

# 점수 설정
CANDIDATE_SCORE_THRESHOLD = 0.25  # 임계값 완화 (recall 향상)
BASE_SCORES = {
    ColorPatternType.SIMPLE: 0.35,
    ColorPatternType.SLASH: 0.40,
    ColorPatternType.COMPOUND: 0.35,
    ColorPatternType.NUMERIC: 0.20,
}


def detect_color_regions_bbox(
    ocr_lines: List[OcrLine],
    image_size: Tuple[int, int],
    valid_colors: Optional[Set[str]] = None
) -> List[Tuple[int, int, List[int]]]:
    """
    색상-수량 영역 후보 감지 (Score-based Candidate Collector v2)

    개선점:
    - ColorRegionCandidate 메타데이터 활용 (내부)
    - 동일 토큰 중복 점수화 방지 (dedupe)
    - 점수 포화 제거 (상대 비교 가능)
    - 아이템 블록 문맥 인식

    Args:
        ocr_lines: OCR 라인 목록
        image_size: 이미지 크기 (width, height)
        valid_colors: DB 색상 목록 (가산점용, 탈락 조건 아님)

    Returns:
        [(y_start, y_end, [line_indices]), ...] - 하위 호환성 유지
    """
    if not ocr_lines:
        return []

    # Phase 0: 아이템 라인 마킹 (블록 문맥용)
    mark_item_lines(ocr_lines)

    # Phase 1: 라인별 점수 계산 (dedupe 적용)
    line_candidates = []
    for i, line in enumerate(ocr_lines):
        candidate = score_color_line(i, line.text, valid_colors)
        if candidate and candidate.score >= CANDIDATE_SCORE_THRESHOLD:
            line.is_color_region = True
            line_candidates.append(candidate)
            logger.debug(
                f"색상 후보: line[{i}] score={candidate.score:.2f} "
                f"tokens={candidate.unique_token_count} text='{line.text[:50]}'"
            )

    if not line_candidates:
        logger.info("색상 후보 라인 없음")
        return []

    logger.info(f"색상 후보 라인: {len(line_candidates)}개 (threshold={CANDIDATE_SCORE_THRESHOLD})")

    # Phase 2: 인접 라인 그룹화 + 블록 문맥
    region_candidates = group_candidate_lines_v2(line_candidates, ocr_lines, image_size)

    logger.info(f"색상 영역 감지: {len(region_candidates)}개 영역")
    for rc in region_candidates:
        logger.debug(
            f"  영역: y={rc.y_start}-{rc.y_end}, lines={rc.line_indices}, "
            f"total_score={rc.total_score:.2f}, ctx={rc.context_type}"
        )

    # 하위 호환: 튜플 반환
    return [(rc.y_start, rc.y_end, rc.line_indices) for rc in region_candidates]


def detect_color_regions_with_metadata(
    ocr_lines: List[OcrLine],
    image_size: Tuple[int, int],
    valid_colors: Optional[Set[str]] = None
) -> List[ColorRegionCandidate]:
    """
    색상 영역 감지 (메타데이터 포함 버전)

    디버깅/분석용 - ColorRegionCandidate 객체 반환
    """
    if not ocr_lines:
        return []

    mark_item_lines(ocr_lines)

    line_candidates = []
    for i, line in enumerate(ocr_lines):
        candidate = score_color_line(i, line.text, valid_colors)
        if candidate and candidate.score >= CANDIDATE_SCORE_THRESHOLD:
            line.is_color_region = True
            line_candidates.append(candidate)

    if not line_candidates:
        return []

    return group_candidate_lines_v2(line_candidates, ocr_lines, image_size)


def mark_item_lines(ocr_lines: List[OcrLine]) -> None:
    """아이템 시작 라인 마킹 (블록 문맥용)"""
    for line in ocr_lines:
        text_upper = line.text.upper()
        for pattern in ITEM_LINE_PATTERNS:
            if re.search(pattern, text_upper):
                line.is_item_line = True
                break


def score_color_line(
    line_index: int,
    text: str,
    valid_colors: Optional[Set[str]] = None
) -> Optional[ColorLineCandidate]:
    """
    라인별 색상 후보 점수 계산 (v2: dedupe + 상대 점수)

    개선점:
    - 동일 토큰 중복 점수화 방지
    - 점수 상한 제거 (상대 비교 가능)
    - 기본 점수 하향 조정
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
    seen_tokens: Set[str] = set()  # dedupe용
    all_color_qty_pairs: List[Tuple[str, int]] = []
    pattern_types: List[ColorPatternType] = []

    # 유형별 패턴 매칭 (dedupe 적용)
    for pattern_type, patterns in COLOR_QTY_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                base_score = BASE_SCORES.get(pattern_type, 0.3)

                for match in matches:
                    color, qty = match[0], int(match[1])
                    token_key = f"{color}:{qty}"

                    # 길이 제한 + dedupe
                    if 1 <= len(color) <= 10 and token_key not in seen_tokens:
                        seen_tokens.add(token_key)
                        score += base_score
                        matched_patterns.append(f"{pattern_type.value}:{color}")
                        all_color_qty_pairs.append((color, qty))
                        if pattern_type not in pattern_types:
                            pattern_types.append(pattern_type)

    if not all_color_qty_pairs:
        return None

    unique_token_count = len(all_color_qty_pairs)

    # 복수 토큰 보너스 (작게 조정)
    if unique_token_count >= 2:
        score += 0.1 * (unique_token_count - 1)  # 토큰당 0.1 추가
        matched_patterns.append(f"bonus:multi_token({unique_token_count})")

    # 라인 길이 보너스
    if len(text_upper) <= 25:
        score += 0.08
    elif len(text_upper) <= 40:
        score += 0.04

    # 가산점 신호
    for pattern, bonus in BONUS_SIGNALS:
        if re.search(pattern, text_upper):
            score += bonus
            matched_patterns.append(f"bonus:{pattern[:15]}")

    # 감점 패턴
    for pattern, penalty in PENALTY_PATTERNS:
        if re.search(pattern, text_upper):
            score += penalty
            matched_patterns.append(f"penalty:{pattern[:15]}")

    # DB 색상 매칭 (가산점)
    db_match_count = 0
    if valid_colors:
        for color, _ in all_color_qty_pairs:
            if is_color_in_db_fuzzy(color, valid_colors):
                db_match_count += 1
                score += 0.12
                matched_patterns.append(f"db_match:{color}")

    # 점수 하한만 적용 (상한 제거)
    score = max(0.0, score)

    return ColorLineCandidate(
        line_index=line_index,
        text=text_upper,
        score=score,
        matched_patterns=matched_patterns,
        color_qty_pairs=all_color_qty_pairs,
        pattern_types=pattern_types,
        db_match_count=db_match_count,
        unique_token_count=unique_token_count
    )


def group_candidate_lines_v2(
    candidates: List[ColorLineCandidate],
    ocr_lines: List[OcrLine],
    image_size: Tuple[int, int]
) -> List[ColorRegionCandidate]:
    """
    후보 라인 그룹화 (v2: 블록 문맥 + 메타데이터)

    개선점:
    - 아이템 라인 경계 인식 (다른 아이템으로 넘어가면 분리)
    - ColorRegionCandidate 메타데이터 반환
    - 문맥 유형 분류
    """
    if not candidates:
        return []

    # 라인 인덱스 순 정렬
    sorted_candidates = sorted(candidates, key=lambda c: c.line_index)

    groups: List[List[ColorLineCandidate]] = []
    current_group = [sorted_candidates[0]]

    for candidate in sorted_candidates[1:]:
        prev_candidate = current_group[-1]
        prev_line = ocr_lines[prev_candidate.line_index]
        curr_line = ocr_lines[candidate.line_index]

        # Y 거리 계산
        y_gap = curr_line.y_top - prev_line.y_bottom

        # 라인 인덱스 거리
        line_gap = candidate.line_index - prev_candidate.line_index - 1

        # 중간에 아이템 라인이 있는지 확인 (블록 경계)
        has_item_boundary = False
        if line_gap > 0:
            for idx in range(prev_candidate.line_index + 1, candidate.line_index):
                if idx < len(ocr_lines) and ocr_lines[idx].is_item_line:
                    has_item_boundary = True
                    break

        # 그룹화 조건:
        # 1. Y 거리 70px 이내
        # 2. 중간 라인 3개 이하
        # 3. 아이템 경계 없음
        if y_gap <= 70 and line_gap <= 3 and not has_item_boundary:
            current_group.append(candidate)
        else:
            groups.append(current_group)
            current_group = [candidate]

    groups.append(current_group)

    # ColorRegionCandidate 생성
    region_candidates = []
    image_height = image_size[1]

    for group in groups:
        line_indices = [c.line_index for c in group]

        # 그룹 내 모든 라인의 Y 범위
        y_start = min(ocr_lines[i].y_top for i in line_indices)
        y_end = max(ocr_lines[i].y_bottom for i in line_indices)

        # 중간 라인 포함
        min_idx = min(line_indices)
        max_idx = max(line_indices)
        expanded_indices = list(range(min_idx, max_idx + 1))

        # 마진 추가
        y_start = max(0, y_start - 25)
        y_end = min(image_height, y_end + 25)

        # 최소 크기 확인
        if y_end - y_start < 25:
            continue

        # 점수 계산
        total_score = sum(c.score for c in group)
        avg_score = total_score / len(group) if group else 0

        # 문맥 유형 결정
        context_type = determine_context_type(min_idx, ocr_lines)

        # raw 텍스트 수집
        raw_texts = [ocr_lines[i].text for i in expanded_indices if i < len(ocr_lines)]

        region_candidates.append(ColorRegionCandidate(
            y_start=y_start,
            y_end=y_end,
            line_indices=expanded_indices,
            total_score=total_score,
            avg_score=avg_score,
            candidate_lines=group,
            raw_line_texts=raw_texts,
            context_type=context_type
        ))

    return region_candidates


def determine_context_type(first_line_idx: int, ocr_lines: List[OcrLine]) -> str:
    """영역의 문맥 유형 결정"""
    # 바로 앞에 아이템 라인이 있는지
    if first_line_idx > 0:
        for i in range(first_line_idx - 1, max(0, first_line_idx - 3) - 1, -1):
            if ocr_lines[i].is_item_line:
                return "after_item"

    # 여러 아이템에 걸쳐 있는지 (rare)
    item_count = sum(1 for line in ocr_lines[:first_line_idx] if line.is_item_line)
    if item_count > 1:
        return "multi_item"

    return "standalone"


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
# 텍스트 병합 (Line-list 기반 v2)
# =============================================================================

def normalize_color_region_text(region_text: str) -> List[str]:
    """
    재OCR된 region_text를 파서 친화적 라인 리스트로 정규화

    문제점:
    - region_text가 여러 줄이 하나로 붙어있을 수 있음
    - 빈 줄, 공백만 있는 줄 처리 필요
    - 한 줄에 여러 색상-수량 토큰이 붙어있을 수 있음

    정규화 규칙:
    1. 줄바꿈 기준 분리
    2. 각 줄 strip (앞뒤 공백 제거)
    3. 빈 줄 제거
    4. 연속 공백 정규화 (2개 이상 → 1개)
    5. **Token-level splitting**: 한 줄에 색상-수량 패턴 2개 이상 시 분리

    Returns:
        정규화된 라인 리스트
    """
    if not region_text:
        return []

    lines = region_text.split('\n')
    normalized = []

    for line in lines:
        # strip 및 연속 공백 정규화
        cleaned = ' '.join(line.split())
        if cleaned:
            # Token-level splitting 시도
            split_lines = split_color_tokens(cleaned)
            normalized.extend(split_lines)

    logger.debug(f"[정규화] 원본 {len(lines)}줄 → 정규화 {len(normalized)}줄")
    for i, line in enumerate(normalized):
        logger.debug(f"  [{i}] '{line[:60]}{'...' if len(line) > 60 else ''}'")

    return normalized


def split_color_tokens(line: str) -> List[str]:
    """
    한 줄에 여러 색상-수량 토큰이 있으면 분리

    예시:
    - "1B - 3 2 - 5 30 - 6" → ["1B - 3", "2 - 5", "30 - 6"]
    - "P4/30 - 2 1B - 3" → ["P4/30 - 2", "1B - 3"]
    - "1B - 3" → ["1B - 3"] (단일 토큰은 그대로)
    - "SOME TEXT" → ["SOME TEXT"] (패턴 없으면 그대로)

    Returns:
        분리된 라인 리스트
    """
    if not line:
        return []

    # 색상-수량 토큰 패턴 (넓은 매칭)
    # 색상: 알파벳/숫자/슬래시 조합 (1-10자)
    # 구분자: - – — (공백 포함 가능)
    # 수량: 1-3자리 숫자
    token_pattern = r'([A-Z0-9][A-Z0-9/]*(?:-[A-Z0-9]+)?)\s*[-–—]\s*(\d{1,3})'

    line_upper = line.upper()
    matches = list(re.finditer(token_pattern, line_upper))

    # 토큰이 2개 미만이면 분리하지 않음
    if len(matches) < 2:
        return [line]

    # 각 토큰을 독립 라인으로
    result = []
    for match in matches:
        color = match.group(1)
        qty = match.group(2)
        # 원본 대소문자 유지를 위해 원본에서 추출
        start, end = match.span()
        original_token = line[start:end] if start < len(line) else f"{color} - {qty}"
        # 정규화된 형태로 저장 (일관성)
        result.append(f"{color} - {qty}")

    logger.debug(f"[Token Split] '{line[:40]}...' → {len(result)}개 토큰")
    return result


def merge_by_replacement(ocr_lines: List[OcrLine], color_results: List[ColorRegionResult]) -> str:
    """
    Line-list 기반 Block Merge (v2)

    개선점:
    - 문자열 교체 → 라인 리스트 치환
    - region_text 정규화 적용
    - 라인별 디버그 로깅

    파서-병합 계약:
    - 아이템 헤더: 1줄
    - 가격 라인: 1줄
    - 색상 라인: 독립적인 줄로 유지
    """
    logger.info(f"[Merge] 시작: {len(ocr_lines)}개 원본 라인, {len(color_results)}개 색상 영역")

    if not color_results:
        result = "\n".join([line.text for line in ocr_lines])
        logger.debug("[Merge] 색상 영역 없음 - 원본 그대로 반환")
        return result

    # 교체 맵 생성 (라인 인덱스 → 대체 라인 리스트)
    replaced_indices: Set[int] = set()
    replacements: Dict[int, List[str]] = {}  # {첫번째_라인_인덱스: [대체_라인들]}

    for result in color_results:
        if result.original_lines:
            first_idx = min(result.original_lines)

            # region_text 정규화 (줄 구조 보존)
            normalized_lines = normalize_color_region_text(result.region_text)

            replacements[first_idx] = normalized_lines

            # 원본 라인들은 모두 제거 대상
            for idx in result.original_lines:
                replaced_indices.add(idx)

            logger.debug(
                f"[Merge] 영역 y={result.y_start}-{result.y_end}: "
                f"원본 라인 {result.original_lines} → {len(normalized_lines)}줄로 교체"
            )

    # 병합 수행 (라인 리스트 기반)
    merged_lines: List[str] = []
    original_used = 0
    replaced_count = 0

    for i, line in enumerate(ocr_lines):
        if i in replacements:
            # 정규화된 라인 리스트로 교체
            replacement_lines = replacements[i]
            merged_lines.extend(replacement_lines)
            replaced_count += len(replacement_lines)

            # 디버그: 교체 상세
            logger.debug(f"[Merge] line[{i}] 교체:")
            logger.debug(f"  원본: '{line.text[:50]}{'...' if len(line.text) > 50 else ''}'")
            for j, rep_line in enumerate(replacement_lines):
                logger.debug(f"  대체[{j}]: '{rep_line[:50]}{'...' if len(rep_line) > 50 else ''}'")

        elif i not in replaced_indices:
            # 교체 대상이 아닌 일반 라인 유지
            merged_lines.append(line.text)
            original_used += 1

        # replaced_indices에만 있는 경우 (중간/끝 라인): 건너뜀

    # 최종 병합 결과 로깅
    logger.info(
        f"[Merge] 완료: 원본 {original_used}줄 유지 + 대체 {replaced_count}줄 "
        f"= 총 {len(merged_lines)}줄"
    )

    # 최종 줄 구조 디버그 (파서 디버깅용)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[Merge] 최종 줄 구조:")
        for i, line in enumerate(merged_lines[:30]):  # 처음 30줄만
            marker = ""
            # 아이템 라인 여부 표시
            if re.search(r'\bS[A-Z]{2,}\d{2,}\b', line.upper()):
                marker = " [ITEM]"
            # 색상-수량 패턴 여부 표시
            elif re.search(r'\b[A-Z0-9/]+\s*[-–—]\s*\d{1,3}\b', line.upper()):
                marker = " [COLOR]"
            logger.debug(f"  [{i:2d}]{marker} '{line[:60]}{'...' if len(line) > 60 else ''}'")

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
