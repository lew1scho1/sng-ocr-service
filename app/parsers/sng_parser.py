"""
SNG (Shake-N-Go) 인보이스 파서 (Block-based Extraction 버전)

전략 변경 (2024-07):
- 줄 단위 순차 FSM → 블록 단위 수집 후 후처리
- 아이템 시작 → 다음 아이템까지 모든 줄 수집
- 색상 줄 누락 복구 가능

역할: 순수 Extraction만 담당
- 라인 분리 및 구조 파싱
- 기본 노이즈 제거 (패턴 매칭 위한 최소한의 정규화)
- Raw 값 반환 (item_code_raw, color_raw, description_raw)

OCR 보정 및 DB 매칭은 Rails ProductMatcher에서 처리:
- 색상 보정 (IB→1B, O→0 등)
- 아이템 코드 보정
- 색상 후보 생성
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

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


@dataclass
class ItemBlock:
    """아이템 블록 (수집 단계)"""
    item_code_raw: str
    description_raw: Optional[str] = None
    unit_price: Optional[float] = None
    qty_ordered: int = 0
    qty_shipped: int = 0
    content_lines: List[str] = field(default_factory=list)  # 색상 줄 포함 모든 후속 줄


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
    Phase 2: 블록 처리 → 라인 아이템 생성

    각 블록에서 색상-수량 패턴 추출
    색상이 없으면 색상 없는 아이템으로 저장
    """
    line_items = []

    # 색상-수량 패턴 (넓은 매칭)
    color_qty_pattern = r'([A-Z0-9][A-Z0-9/]*(?:-[A-Z0-9]+)?)\s*[-–—]\s*(\d{1,3})'

    for block in blocks:
        colors_found = []

        # 블록 내 모든 줄에서 색상-수량 추출
        for content_line in block.content_lines:
            matches = re.findall(color_qty_pattern, content_line)
            for match in matches:
                color_raw = match[0].strip()
                quantity = int(match[1])

                if is_valid_color(color_raw):
                    colors_found.append((color_raw, quantity))

        if colors_found:
            # 색상별 라인 아이템 생성
            for color_raw, quantity in colors_found:
                line_items.append(SngLineItem(
                    item_code_raw=block.item_code_raw,
                    color_raw=color_raw,
                    quantity=quantity,
                    unit_price=block.unit_price,
                    description_raw=block.description_raw,
                    qty_ordered=block.qty_ordered,
                    qty_shipped=block.qty_shipped
                ))
                logger.debug(f"색상-수량: {block.item_code_raw} - {color_raw} x {quantity}")
        else:
            # 색상 없는 아이템
            line_items.append(SngLineItem(
                item_code_raw=block.item_code_raw,
                color_raw="",
                quantity=block.qty_shipped,
                unit_price=block.unit_price,
                description_raw=block.description_raw,
                qty_ordered=block.qty_ordered,
                qty_shipped=block.qty_shipped
            ))
            logger.info(f"색상 없는 아이템: {block.item_code_raw}")

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
                "qty_shipped": item.qty_shipped
            }
            for item in result.line_items
        ],
        "raw_text_preview": result.raw_text,
        "last_item_code_raw": result.last_item_code_raw,
        "last_unit_price": result.last_unit_price
    }
