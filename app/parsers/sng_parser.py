"""
SNG (Shake-N-Go) 인보이스 파서 (2-Pass Block Processing 버전)

전략 변경 (2024-07):
- 1-pass 확정 판정 → 2-pass 후보 수집 + 후처리
- 확정 색상 / 약한 후보 / 기타 줄 분리
- 색상 판정 실패 후보도 raw candidate로 보존

핵심 원칙:
- "확정 실패"와 "후보 폐기"를 분리
- early discard 방지
- raw candidate를 block 후단까지 보존

역할: 순수 Extraction만 담당
- 라인 분리 및 구조 파싱
- 기본 노이즈 제거 (패턴 매칭 위한 최소한의 정규화)
- Raw 값 반환 (item_code_raw, color_raw, description_raw)
- raw_candidates 포함 (후단 분석용)

OCR 보정 및 DB 매칭은 Rails ProductMatcher에서 처리:
- 색상 보정 (IB→1B, O→0 등)
- 아이템 코드 보정
- 색상 후보 생성
"""

import re
import logging
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# 수량 OCR 오인식 매핑 (보편적)
QTY_OCR_MAP = {
    'l': '1', 'L': '1', 'I': '1', 'i': '1',
    'e': '6', 'E': '6',
    'o': '0', 'O': '0', 'C': '0', 'c': '0',
    'q': '4', 'Q': '4',
    'a': '4', 'A': '4',
    ')': '', '(': '', ' ': ''
}


class CandidateStatus(Enum):
    """색상 후보 상태"""
    CONFIRMED = "confirmed"        # DB/패턴상 확정
    WEAK = "weak"                  # 구조는 색상 같지만 OCR 오류 가능
    REJECTED = "rejected"          # 명확히 비색상


@dataclass
class RawColorCandidate:
    """색상 후보 (raw 데이터 보존)"""
    color_raw: str                      # OCR 원본 색상 토큰
    qty_raw: str                        # OCR 원본 수량 토큰
    color_normalized: Optional[str] = None   # 정규화된 색상 (가능하면)
    qty_normalized: Optional[int] = None     # 정규화된 수량 (가능하면)
    source_line: str = ""               # 출처 라인
    status: CandidateStatus = CandidateStatus.WEAK
    score: float = 0.0                  # 후보 점수
    reason_flags: List[str] = field(default_factory=list)  # 판정 이유


@dataclass
class ItemBlock:
    """아이템 블록 (2-pass 수집 단계)"""
    item_code_raw: str
    description_raw: Optional[str] = None
    unit_price: Optional[float] = None
    qty_ordered: int = 0
    qty_shipped: int = 0
    content_lines: List[str] = field(default_factory=list)  # 모든 후속 줄
    # 2-pass용 필드
    confirmed_colors: List[RawColorCandidate] = field(default_factory=list)
    raw_color_candidates: List[RawColorCandidate] = field(default_factory=list)
    unclassified_lines: List[str] = field(default_factory=list)


@dataclass
class SngInvoiceHeader:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None


@dataclass
class SngLineItem:
    """라인 아이템 (raw 데이터 - 보정 없음)"""
    item_code_raw: str              # OCR 원본 아이템 코드
    color_raw: str                  # OCR 원본 색상 (보정 없음, Rails에서 처리)
    quantity: int                   # 수량 (파싱용 정수 변환만 적용)
    unit_price: Optional[float] = None
    description_raw: Optional[str] = None  # OCR 원본 설명
    qty_ordered: Optional[int] = None
    qty_shipped: Optional[int] = None
    # 2-pass용 디버그 필드
    color_status: str = "confirmed"     # "confirmed" | "weak" | "no_color"
    raw_candidates: List[dict] = field(default_factory=list)  # 디버그용 raw 후보들


@dataclass
class SngInvoiceResult:
    header: SngInvoiceHeader
    line_items: List[SngLineItem]
    raw_text: str
    # 멀티 페이지 지원
    last_item_code_raw: Optional[str] = None
    last_unit_price: Optional[float] = None


def normalize_line(line: str) -> str:
    """
    기본 OCR 노이즈 제거

    1. 특수문자 제거: « »
    2. 0G → OG (설명 코드)
    3. 쉼표 → 점 (가격)
    4. 가격 OCR 보정: .08/.80/.88 → .00
    5. 공백 정규화
    """
    # 1. 특수문자 제거
    line = re.sub(r'[«»]', '', line)

    # 2. 0G → OG
    line = re.sub(r'\(0G\b', 'OG', line)
    line = re.sub(r'\b0G\b', 'OG', line)

    # 3. 가격 내 쉼표 → 점
    line = re.sub(r'(\d+),(\d{2})\b', r'\1.\2', line)

    # 4. 가격 OCR 보정 (0이 8로 오인식)
    def fix_price_ocr(match):
        price = match.group(0)
        cents = price[-2:]
        if cents in ('08', '80', '88'):
            return price[:-2] + '00'
        return price
    line = re.sub(r'\d+\.\d{2}\b', fix_price_ocr, line)

    # 5. 공백 정규화
    line = re.sub(r'\s+', ' ', line)

    return line.strip()


def normalize_qty_string(s: str) -> Optional[int]:
    """수량 OCR 보정"""
    if not s or not s.strip():
        return None

    cleaned = ''.join(QTY_OCR_MAP.get(c, c) for c in s.strip())

    if not cleaned:
        return 0

    try:
        return int(cleaned)
    except ValueError:
        logger.debug(f"수량 변환 실패: '{s}' → '{cleaned}'")
        return None


def parse_sng_invoice(text: str, prev_item_code: Optional[str] = None,
                      prev_unit_price: Optional[float] = None) -> SngInvoiceResult:
    """SNG 인보이스 파싱"""
    header = extract_header(text)
    line_items = extract_line_items(text, prev_item_code, prev_unit_price)

    # 마지막 아이템 정보 (다음 페이지 연결용)
    last_item_code_raw = None
    last_unit_price = None
    if line_items:
        last_item = line_items[-1]
        last_item_code_raw = last_item.item_code_raw
        last_unit_price = last_item.unit_price

    return SngInvoiceResult(
        header=header,
        line_items=line_items,
        raw_text=text[:1000] if text else "",
        last_item_code_raw=last_item_code_raw,
        last_unit_price=last_unit_price
    )


def extract_header(text: str) -> SngInvoiceHeader:
    """헤더 정보 추출"""
    header = SngInvoiceHeader()

    # 날짜 OCR 노이즈 보정
    normalized_text = re.sub(r'(\d{1,2})/[€$@#](\d)/(\d{4})', r'\1/0\2/\3', text)

    # Invoice NO (10자리 숫자)
    invoice_no_patterns = [
        r'Invoice\s*(?:NO\.?|#)\s*(\d{10})',
        r'(\d{10})',
    ]

    for pattern in invoice_no_patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            header.invoice_number = match.group(1)
            break

    # Invoice Date
    lines = normalized_text.split('\n')

    for i, line in enumerate(lines[:25]):
        if 'INVOICE' in line.upper() and 'DATE' in line.upper():
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            if date_match:
                header.invoice_date = date_match.group(1)
                break
            if i + 1 < len(lines):
                date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', lines[i + 1])
                if date_match:
                    header.invoice_date = date_match.group(1)
                    break

    if not header.invoice_date:
        for line in lines[:10]:
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            if date_match:
                header.invoice_date = date_match.group(1)
                break

    return header


def extract_line_items(text: str, prev_item_code: Optional[str] = None,
                       prev_unit_price: Optional[float] = None) -> List[SngLineItem]:
    """
    블록 기반 라인 아이템 추출 (raw 데이터)

    2단계 처리:
    1. 블록 수집: 아이템 시작 → 다음 아이템까지 모든 줄 수집
    2. 블록 처리: 각 블록에서 색상-수량 추출

    이점:
    - 색상 줄 누락 시 복구 가능
    - 아이템 경계 명확화
    """
    lines = text.split('\n')

    # Phase 1: 블록 수집
    blocks = collect_item_blocks(lines, prev_item_code, prev_unit_price)
    logger.info(f"수집된 아이템 블록: {len(blocks)}개")

    # Phase 2: 블록 처리 → 라인 아이템 생성
    line_items = process_item_blocks(blocks)

    logger.info(f"추출된 라인 아이템: {len(line_items)}")
    return line_items


def collect_item_blocks(lines: List[str], prev_item_code: Optional[str] = None,
                        prev_unit_price: Optional[float] = None) -> List[ItemBlock]:
    """
    Phase 1: 아이템 블록 수집

    아이템 코드 패턴을 찾아 블록 경계 설정
    각 블록에 후속 줄들을 모두 수집
    """
    blocks = []
    current_block: Optional[ItemBlock] = None

    # 이전 페이지 연결
    if prev_item_code:
        current_block = ItemBlock(
            item_code_raw=prev_item_code,
            unit_price=prev_unit_price
        )

    # 아이템 코드 감지 패턴들 (넓은 매칭)
    qty_char_class = r'[0-9leoqaciLEOQACIO\)\(]+'

    # 패턴 1: 수량 + 아이템코드 + 설명코드
    item_pattern_with_qty = rf'({qty_char_class})\s+({qty_char_class})\s+(S[A-Za-z0-9]{{4,8}})\s+(OG|HR|0G)\b'

    # 패턴 2: 아이템코드 + 설명코드 (수량 없이)
    item_pattern_simple = r'\b(S[A-Za-z0-9]{5,8})\s+(OG|HR|0G)\b'

    # 패턴 3: 아이템코드만 (S로 시작, 6-8자)
    item_pattern_code_only = r'\b(S[A-Z0-9]{5,7})\b'

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue

        normalized_line = normalize_line(original_line)
        line_upper = normalized_line.upper()

        # 헤더 행 제외
        if re.search(r'\b(PACKED|ORDERED|SHIPPED|DESCRIPTION|PRICE|EXTENDED|ITEM\s*NUMBER)\b', line_upper):
            continue

        # 아이템 코드 감지 시도
        item_info = detect_item_start(line_upper, item_pattern_with_qty, item_pattern_simple, item_pattern_code_only)

        if item_info:
            # 이전 블록 저장
            if current_block:
                blocks.append(current_block)

            # 새 블록 시작
            item_code, desc_raw, qty_ordered, qty_shipped = item_info
            current_block = ItemBlock(
                item_code_raw=item_code,
                description_raw=desc_raw,
                qty_ordered=qty_ordered,
                qty_shipped=qty_shipped
            )

            # 같은 줄에서 가격 추출
            price_match = re.search(r'(\d{1,3}\.\d{2})', normalized_line)
            if price_match:
                current_block.unit_price = float(price_match.group(1))

            logger.info(f"블록 시작: {item_code} (Qty: {qty_ordered}/{qty_shipped})")
        else:
            # 현재 블록에 줄 추가
            if current_block:
                current_block.content_lines.append(line_upper)

                # 가격 추출 (아직 없으면)
                if current_block.unit_price is None:
                    price_match = re.search(r'(\d{1,3}\.\d{2})', normalized_line)
                    if price_match:
                        current_block.unit_price = float(price_match.group(1))

    # 마지막 블록 저장
    if current_block:
        blocks.append(current_block)

    return blocks


def detect_item_start(line_upper: str, pattern_with_qty: str, pattern_simple: str,
                      pattern_code_only: str) -> Optional[Tuple[str, Optional[str], int, int]]:
    """
    아이템 시작 줄 감지

    Returns:
        (item_code, description_raw, qty_ordered, qty_shipped) or None
    """
    # 패턴 1: 수량 포함
    match = re.search(pattern_with_qty, line_upper)
    if match:
        qty1_raw = match.group(1)
        qty2_raw = match.group(2)
        item_code = match.group(3).upper()
        desc_prefix = match.group(4).upper()

        if item_code.startswith('S'):
            order_qty = normalize_qty_string(qty1_raw) or 0
            ship_qty = normalize_qty_string(qty2_raw) or 0

            # Description 추출
            desc_match = re.search(
                rf'{re.escape(desc_prefix)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)',
                line_upper
            )
            if desc_match:
                desc_raw = f"{desc_prefix} {desc_match.group(1).strip()}"
            else:
                desc_raw = desc_prefix

            return (item_code, desc_raw, order_qty, ship_qty)

    # 패턴 2: 아이템코드 + 설명코드
    match = re.search(pattern_simple, line_upper)
    if match:
        item_code = match.group(1).upper()
        desc_prefix = match.group(2).upper()

        if item_code.startswith('S'):
            # Description 추출
            desc_match = re.search(
                rf'{re.escape(desc_prefix)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)',
                line_upper
            )
            if desc_match:
                desc_raw = f"{desc_prefix} {desc_match.group(1).strip()}"
            else:
                desc_raw = desc_prefix

            return (item_code, desc_raw, 0, 0)

    # 패턴 3: 아이템코드만 (엄격한 조건)
    # - 가격이 같은 줄에 있어야 함
    # - SO로 시작해야 함 (SNG 아이템 코드 특성)
    # - 숫자를 포함해야 함 (STANDARD 같은 단어 제외)
    if re.search(r'\d{1,3}\.\d{2}', line_upper):
        match = re.search(pattern_code_only, line_upper)
        if match:
            item_code = match.group(1).upper()
            # SO로 시작하고 숫자 포함
            if item_code.startswith('SO') and re.search(r'\d', item_code):
                return (item_code, None, 0, 0)

    return None


def process_item_blocks(blocks: List[ItemBlock]) -> List[SngLineItem]:
    """
    Phase 2: 2-pass 블록 처리 → 라인 아이템 생성

    Pass 1: 느슨한 패턴으로 모든 후보 수집 (아무것도 버리지 않음)
    Pass 2: 후보 분류 및 최종 판정

    핵심 원칙:
    - "확정 실패"와 "후보 폐기"를 분리
    - confirmed color가 없어도 raw candidate가 있으면 보존
    - 둘 다 없을 때만 진짜 "색상 없는 아이템"
    """
    line_items = []

    for block in blocks:
        # Pass 1: 후보 수집 (느슨한 패턴)
        collect_color_candidates(block)

        # Pass 2: 후보 분류 및 라인 아이템 생성
        items = finalize_block_items(block)
        line_items.extend(items)

    return line_items


def collect_color_candidates(block: ItemBlock) -> None:
    """
    Pass 1: 느슨한 패턴으로 색상 후보 수집

    수집 규칙:
    - 짧은 토큰 + 숫자 구조면 우선 보존
    - OCR 오류가 낀 수량도 raw로 보존
    - 정확한 color code가 아니라 color-like + qty-like 구조면 수집

    느슨한 후보 패턴:
    - XX - 3, 613 - 2 (짧은 토큰 뒤 숫자)
    - P4/30 - 2 (slash 포함)
    - C-42730 - 4, C 42730 - 4 (compound)
    - 30 - G, 1B - S (OCR 오류 수량)
    """
    # 느슨한 색상-수량 패턴들
    # 패턴 1: 정상 색상-수량 (확정 후보)
    confirmed_pattern = r'([A-Z0-9][A-Z0-9/]*(?:-[A-Z0-9]+)?)\s*[-–—]\s*(\d{1,3})'

    # 패턴 2: 수량이 OCR 오류인 경우 (약한 후보)
    # 예: 30 - G, 1B - S, P4/30 - Z
    weak_qty_pattern = r'([A-Z0-9][A-Z0-9/]*(?:-[A-Z0-9]+)?)\s*[-–—]\s*([A-Z0-9]{1,3})'

    # 패턴 3: 복합 색상 (공백 포함) - C 42730 - 4
    compound_pattern = r'([A-Z])\s+(\d{4,6})\s*[-–—]\s*(\d{1,3})'

    # 제외 패턴 (명확한 비색상)
    exclude_patterns = [
        r'\b(WATER|BODY|WAVE|CURL|TWIST|BRAID|DEEP|LOOSE|NATURAL|ORGANIQUE)\b',
        r'\b\d{1,3}\.\d{2}\b',  # 가격
        r'\bS[A-Z]{4,}\d*\b',   # 아이템 코드
    ]

    for content_line in block.content_lines:
        line_upper = content_line.upper().strip()

        # 명확한 비색상 줄 분류
        is_non_color = False
        for pattern in exclude_patterns:
            if re.search(pattern, line_upper):
                # 제외 패턴이 있어도 색상 패턴도 있으면 후보로 유지
                if not re.search(confirmed_pattern, line_upper):
                    is_non_color = True
                    break

        if is_non_color and not re.search(weak_qty_pattern, line_upper):
            block.unclassified_lines.append(line_upper)
            continue

        # 패턴 1: 확정 후보 수집
        matches = re.findall(confirmed_pattern, line_upper)
        for match in matches:
            color_raw = match[0].strip()
            qty_raw = match[1].strip()

            if len(color_raw) < 1 or len(color_raw) > 15:
                continue

            # 수량 정규화 시도
            qty_normalized = normalize_qty_string(qty_raw)

            candidate = RawColorCandidate(
                color_raw=color_raw,
                qty_raw=qty_raw,
                color_normalized=color_raw,  # 색상은 그대로
                qty_normalized=qty_normalized,
                source_line=line_upper,
                status=CandidateStatus.CONFIRMED if qty_normalized else CandidateStatus.WEAK,
                score=1.0 if qty_normalized else 0.7,
                reason_flags=["pattern:confirmed"] if qty_normalized else ["pattern:qty_parse_weak"]
            )

            if candidate.status == CandidateStatus.CONFIRMED:
                block.confirmed_colors.append(candidate)
            else:
                block.raw_color_candidates.append(candidate)

        # 패턴 2: 약한 후보 수집 (수량이 문자인 경우)
        weak_matches = re.findall(weak_qty_pattern, line_upper)
        for match in weak_matches:
            color_raw = match[0].strip()
            qty_raw = match[1].strip()

            # 이미 확정 후보에 있으면 건너뜀
            if any(c.color_raw == color_raw and c.qty_raw == qty_raw
                   for c in block.confirmed_colors + block.raw_color_candidates):
                continue

            if len(color_raw) < 1 or len(color_raw) > 15:
                continue

            # 순수 숫자가 아닌 경우만 (이미 위에서 처리됨)
            if qty_raw.isdigit():
                continue

            # OCR 오류 수량 정규화 시도
            qty_normalized = normalize_qty_string(qty_raw)

            candidate = RawColorCandidate(
                color_raw=color_raw,
                qty_raw=qty_raw,
                color_normalized=color_raw,
                qty_normalized=qty_normalized,
                source_line=line_upper,
                status=CandidateStatus.WEAK,
                score=0.5,
                reason_flags=["pattern:weak_qty", f"qty_raw:{qty_raw}"]
            )
            block.raw_color_candidates.append(candidate)

        # 패턴 3: 복합 색상 수집
        compound_matches = re.findall(compound_pattern, line_upper)
        for match in compound_matches:
            prefix = match[0].strip()
            number = match[1].strip()
            qty_raw = match[2].strip()

            color_raw = f"{prefix}-{number}"

            # 이미 있으면 건너뜀
            if any(c.color_raw == color_raw for c in block.confirmed_colors + block.raw_color_candidates):
                continue

            qty_normalized = normalize_qty_string(qty_raw)

            candidate = RawColorCandidate(
                color_raw=color_raw,
                qty_raw=qty_raw,
                color_normalized=color_raw,
                qty_normalized=qty_normalized,
                source_line=line_upper,
                status=CandidateStatus.CONFIRMED if qty_normalized else CandidateStatus.WEAK,
                score=0.9 if qty_normalized else 0.6,
                reason_flags=["pattern:compound"]
            )

            if candidate.status == CandidateStatus.CONFIRMED:
                block.confirmed_colors.append(candidate)
            else:
                block.raw_color_candidates.append(candidate)

    # 로깅
    logger.debug(
        f"[Pass1] {block.item_code_raw}: "
        f"confirmed={len(block.confirmed_colors)}, "
        f"weak={len(block.raw_color_candidates)}, "
        f"unclassified={len(block.unclassified_lines)}"
    )


def finalize_block_items(block: ItemBlock) -> List[SngLineItem]:
    """
    Pass 2: 후보 분류 및 최종 라인 아이템 생성

    판정 규칙:
    1. confirmed color가 있으면 → 일반 처리
    2. confirmed는 없지만 raw candidate가 있으면 → "색상 미확정 후보 포함" 아이템
    3. 둘 다 없을 때만 → 진짜 "색상 없는 아이템"
    """
    line_items = []

    # 디버그용 raw candidates 직렬화
    all_candidates_dict = [
        {
            "color_raw": c.color_raw,
            "qty_raw": c.qty_raw,
            "color_norm": c.color_normalized,
            "qty_norm": c.qty_normalized,
            "status": c.status.value,
            "score": c.score,
            "reasons": c.reason_flags
        }
        for c in block.confirmed_colors + block.raw_color_candidates
    ]

    # Case 1: 확정 색상이 있음
    if block.confirmed_colors:
        for candidate in block.confirmed_colors:
            qty = candidate.qty_normalized if candidate.qty_normalized else 0
            line_items.append(SngLineItem(
                item_code_raw=block.item_code_raw,
                color_raw=candidate.color_raw,
                quantity=qty,
                unit_price=block.unit_price,
                description_raw=block.description_raw,
                qty_ordered=block.qty_ordered,
                qty_shipped=block.qty_shipped,
                color_status="confirmed",
                raw_candidates=all_candidates_dict
            ))
            logger.debug(f"[확정] {block.item_code_raw} - {candidate.color_raw} x {qty}")

    # Case 2: 확정은 없지만 약한 후보가 있음
    elif block.raw_color_candidates:
        # 점수 높은 순으로 정렬
        sorted_candidates = sorted(block.raw_color_candidates, key=lambda c: c.score, reverse=True)

        for candidate in sorted_candidates:
            qty = candidate.qty_normalized if candidate.qty_normalized else 0
            line_items.append(SngLineItem(
                item_code_raw=block.item_code_raw,
                color_raw=candidate.color_raw,
                quantity=qty,
                unit_price=block.unit_price,
                description_raw=block.description_raw,
                qty_ordered=block.qty_ordered,
                qty_shipped=block.qty_shipped,
                color_status="weak",
                raw_candidates=all_candidates_dict
            ))
            logger.info(
                f"[약한후보] {block.item_code_raw} - {candidate.color_raw} x {qty} "
                f"(score={candidate.score:.2f}, reasons={candidate.reason_flags})"
            )

    # Case 3: 둘 다 없음 → 진짜 색상 없는 아이템
    else:
        line_items.append(SngLineItem(
            item_code_raw=block.item_code_raw,
            color_raw="",
            quantity=block.qty_shipped,
            unit_price=block.unit_price,
            description_raw=block.description_raw,
            qty_ordered=block.qty_ordered,
            qty_shipped=block.qty_shipped,
            color_status="no_color",
            raw_candidates=all_candidates_dict
        ))
        logger.info(
            f"[색상없음] {block.item_code_raw}: "
            f"confirmed=0, weak=0, unclassified={len(block.unclassified_lines)}"
        )

    return line_items


def is_valid_color(color: str) -> bool:
    """유효한 색상 코드 확인"""
    if len(color) < 1 or len(color) > 20:
        return False

    if color.isdigit():
        return True

    if re.match(r'^[A-Z0-9][A-Z0-9/\-]*$', color, re.IGNORECASE):
        return True

    return False


def to_dict(result: SngInvoiceResult) -> dict:
    """JSON 직렬화"""
    return {
        "header": {
            "invoice_number": result.header.invoice_number,
            "invoice_date": result.header.invoice_date
        },
        "line_items": [
            {
                "item_code_raw": item.item_code_raw,
                "color_raw": item.color_raw,  # raw 값 그대로
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "description_raw": item.description_raw,
                "qty_ordered": item.qty_ordered,
                "qty_shipped": item.qty_shipped,
                # 2-pass 디버그 정보
                "color_status": item.color_status,
                "raw_candidates": item.raw_candidates
            }
            for item in result.line_items
        ],
        "raw_text_preview": result.raw_text,
        "last_item_code_raw": result.last_item_code_raw,
        "last_unit_price": result.last_unit_price
    }
