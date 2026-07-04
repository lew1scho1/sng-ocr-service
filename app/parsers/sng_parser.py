"""
SNG (Shake-N-Go) 인보이스 파서 (단순화 버전)

역할: Extraction + Line Segmentation + 보편적 OCR 보정
- 라인 분리 및 구조 파싱 (2줄 가격, 4열 색상-수량)
- 기본 노이즈 제거
- 색상 OCR 보정 (IB→1B)
- 수량 OCR 보정

item_code 보정 및 DB 매칭은 Rails에서 처리
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum, auto

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


class ParserState(Enum):
    """파서 상태 머신"""
    IDLE = auto()           # 아이템 찾는 중
    ITEM_START = auto()     # 아이템 시작 (첫 줄 가격)
    PRICE_LINE = auto()     # 가격 둘째 줄 (Your Price)
    COLOR_COLLECT = auto()  # 색상-수량 수집 중


@dataclass
class SngInvoiceHeader:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None


@dataclass
class SngLineItem:
    """라인 아이템 (raw 데이터)"""
    item_code_raw: str              # OCR 원본 아이템 코드 (보정 없음)
    color: str                      # 색상 (IB→1B 보정만 적용)
    quantity: int                   # 수량
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


def normalize_color_code(color: str) -> str:
    """
    색상 코드 OCR 보정 (보편적)

    IB → 1B, I → 1
    """
    color = color.upper().strip()

    # IB → 1B
    if color.startswith('I') and len(color) >= 2 and color[1] == 'B':
        color = '1' + color[1:]
    # 단독 I → 1
    elif color == 'I':
        color = '1'

    return color


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
    라인 아이템 추출 (raw 데이터)

    item_code는 보정 없이 원본 그대로 반환
    """
    line_items = []
    lines = text.split('\n')

    # 현재 아이템 컨텍스트
    current_item_code_raw = prev_item_code
    current_description_raw = None
    current_unit_price = prev_unit_price
    current_qty_ordered = 0
    current_qty_shipped = 0
    current_has_colors = False
    state = ParserState.COLOR_COLLECT if prev_item_code else ParserState.IDLE

    # 패턴
    qty_char_class = r'[0-9leoqaciLEOQACIO\)\(]+'
    item_line_pattern = rf'(?:^|[A-Z]{{2,3}}\s+)({qty_char_class})\s+({qty_char_class})\s+(S[A-Za-z0-9]{{4,8}})\s+(OG|HR|0G)\b'
    item_code_only_pattern = r'\b(S[A-Za-z0-9]{5,8})\s+(OG|HR|0G)\s+(.+?)\s+(\d{1,3}\.\d{2})'
    color_qty_pattern = r'([A-Z0-9][A-Z0-9/]*(?:-[A-Z0-9]+)?)\s*[-–—]\s*(\d{1,3})(?:\s*\((\d+)\))?'
    price_pattern = r'(\d{1,3}\.\d{2})'

    def save_item_if_no_colors():
        """색상 없는 아이템 저장"""
        nonlocal current_has_colors
        if current_item_code_raw and not current_has_colors:
            line_items.append(SngLineItem(
                item_code_raw=current_item_code_raw,
                color="",
                quantity=current_qty_shipped,
                unit_price=current_unit_price,
                description_raw=current_description_raw,
                qty_ordered=current_qty_ordered,
                qty_shipped=current_qty_shipped
            ))
            logger.info(f"색상 없는 아이템: {current_item_code_raw}, qty={current_qty_shipped}")

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue

        normalized_line = normalize_line(original_line)
        line_upper = normalized_line.upper()

        # 헤더 행 제외
        if re.search(r'\b(PACKED|ORDERED|SHIPPED|DESCRIPTION|PRICE|EXTENDED|ITEM\s*NUMBER)\b', line_upper):
            continue

        # 아이템 코드 찾기 - 1차: 수량 포함 패턴
        item_match = re.search(item_line_pattern, line_upper)
        if item_match:
            qty1_raw = item_match.group(1)
            qty2_raw = item_match.group(2)
            potential_item = item_match.group(3).upper()
            desc_prefix = item_match.group(4).upper()

            order_qty = normalize_qty_string(qty1_raw) or 0
            ship_qty = normalize_qty_string(qty2_raw) or 0

            if potential_item.startswith('S'):
                save_item_if_no_colors()

                # Description 추출
                desc_match = re.search(
                    rf'{re.escape(desc_prefix)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)',
                    line_upper
                )
                if desc_match:
                    current_description_raw = f"{desc_prefix} {desc_match.group(1).strip()}"
                else:
                    current_description_raw = desc_prefix

                # 새 아이템 시작 (raw 코드 그대로)
                current_item_code_raw = potential_item
                current_unit_price = None
                current_qty_ordered = order_qty
                current_qty_shipped = ship_qty
                current_has_colors = False
                state = ParserState.ITEM_START

                logger.info(f"아이템 감지: {potential_item} (Qty: {order_qty}/{ship_qty})")
                continue

        # 2차: 대체 패턴
        if not item_match:
            alt_match = re.search(item_code_only_pattern, line_upper)
            if alt_match:
                potential_item = alt_match.group(1).upper()
                desc_prefix = alt_match.group(2).upper()
                desc_rest = alt_match.group(3).strip()

                if potential_item.startswith('S'):
                    save_item_if_no_colors()

                    current_item_code_raw = potential_item
                    current_description_raw = f"{desc_prefix} {desc_rest}"
                    current_unit_price = None
                    current_qty_ordered = 0
                    current_qty_shipped = 0
                    current_has_colors = False
                    state = ParserState.ITEM_START

                    logger.info(f"아이템 감지 (대체): {potential_item}")
                    continue

        # 가격 추출
        if state == ParserState.ITEM_START and current_item_code_raw:
            prices = re.findall(price_pattern, normalized_line)
            if prices:
                current_unit_price = float(prices[0])
                logger.info(f"가격: {current_unit_price}")
                state = ParserState.COLOR_COLLECT

        # 색상-수량 추출
        if current_item_code_raw:
            color_qty_matches = re.findall(color_qty_pattern, line_upper)

            for match in color_qty_matches:
                raw_color = match[0].strip()
                quantity = int(match[1])

                # 색상 OCR 보정 (보편적: IB→1B)
                color = normalize_color_code(raw_color)

                if is_valid_color(color):
                    line_items.append(SngLineItem(
                        item_code_raw=current_item_code_raw,
                        color=color,
                        quantity=quantity,
                        unit_price=current_unit_price,
                        description_raw=current_description_raw,
                        qty_ordered=current_qty_ordered,
                        qty_shipped=current_qty_shipped
                    ))
                    current_has_colors = True
                    logger.debug(f"색상-수량: {current_item_code_raw} - {color} x {quantity}")

            if color_qty_matches:
                state = ParserState.COLOR_COLLECT

    # 마지막 아이템
    save_item_if_no_colors()

    logger.info(f"추출된 라인 아이템: {len(line_items)}")
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
                "color": item.color,
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
