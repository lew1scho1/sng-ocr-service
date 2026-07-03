"""
SNG (Shake-N-Go) 인보이스 파서

인보이스 구조:
- 헤더: Invoice NO, Invoice Date
- 라인 아이템: ITEM NUMBER, Description, 색상-수량 쌍 (4열), 단가
- 색상-수량 쌍은 여러 줄에 걸쳐 4열로 배치됨

SNG 아이템 코드 형식:
- S로 시작 (SFTWB14, SODWX24, SOHWXL3 등)
- 5-8자리
- OCR 오인식 보정: I→1, O→0, Y→4

가격 구조 (2줄):
- 첫 줄: List Price / List Extended
- 둘째 줄: Your Price / Your Extended / Discounted Amount
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)


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
    item_code: str
    color: str
    quantity: int
    unit_price: Optional[float] = None
    description: Optional[str] = None


@dataclass
class SngInvoiceResult:
    header: SngInvoiceHeader
    line_items: List[SngLineItem]
    raw_text: str
    # 멀티 페이지 지원: 다음 페이지 처리 시 사용
    last_item_code: Optional[str] = None
    last_unit_price: Optional[float] = None


def normalize_line(line: str) -> str:
    """
    OCR 오류를 정규화하여 파싱 가능한 형태로 변환

    적용 순서가 중요:
    1. 특수문자 제거: « » 등
    2. 0/O 보정: 아이템 코드 뒤 설명 코드에서 0G → OG
    3. 쉼표 보정: 가격의 74,08 → 74.08
    4. 가격 0→8 오인식 보정: .08 → .00, .88 → .00
    5. 공백 정규화: 다중 공백 → 단일 공백
    """
    # 1. 특수문자 제거
    line = re.sub(r'[«»]', '', line)

    # 2. 아이템 코드 뒤 설명 코드에서 0G → OG, (0G → OG
    line = re.sub(r'\(0G\b', 'OG', line)
    line = re.sub(r'\b0G\b', 'OG', line)

    # 3. 가격 내 쉼표를 점으로 변환
    line = re.sub(r'(\d+),(\d{2})\b', r'\1.\2', line)

    # 4. 가격 OCR 오류 보정 (0이 8로 오인식)
    # SNG 가격은 .00으로 끝남: 17.08 → 17.00, 32.88 → 32.00
    def fix_price_ocr(match):
        price = match.group(0)
        cents = price[-2:]
        if cents in ('08', '80', '88'):
            return price[:-2] + '00'
        return price
    line = re.sub(r'\d+\.\d{2}\b', fix_price_ocr, line)

    # 5. 다중 공백 정규화
    line = re.sub(r'\s+', ' ', line)

    return line.strip()


def correct_ocr_item_code(code: str) -> str:
    """
    OCR 오인식 문자를 보정하여 올바른 아이템 코드 반환

    예: SFTWBIY → SFTWB14
        soDwx24 → SODWX24
    """
    code = code.upper()
    result = list(code)

    # 끝에서 1-2자리가 숫자여야 함
    # Y→4, I→1, O→0 보정
    ocr_to_digit = {'Y': '4', 'I': '1', 'O': '0'}

    for i in range(len(result) - 1, max(len(result) - 3, 3), -1):
        char = result[i]
        if char in ocr_to_digit:
            result[i] = ocr_to_digit[char]
            logger.debug(f"OCR 보정: {code}[{i}] '{char}' → '{result[i]}'")

    corrected = ''.join(result)
    if corrected != code:
        logger.info(f"OCR 아이템 코드 보정: {code} → {corrected}")

    return corrected


def parse_sng_invoice(text: str, prev_item_code: Optional[str] = None,
                      prev_unit_price: Optional[float] = None) -> SngInvoiceResult:
    """
    SNG 인보이스 OCR 텍스트를 파싱하여 구조화된 데이터 반환

    Args:
        text: OCR 텍스트
        prev_item_code: 이전 페이지의 마지막 아이템 코드 (멀티 페이지 지원)
        prev_unit_price: 이전 페이지의 마지막 단가

    Returns:
        SngInvoiceResult: 파싱 결과 (last_item_code, last_unit_price 포함)
    """
    header = extract_header(text)
    line_items = extract_line_items(text, prev_item_code, prev_unit_price)

    # 마지막 아이템 정보 추출 (다음 페이지 연결용)
    last_item_code = None
    last_unit_price = None
    if line_items:
        last_item = line_items[-1]
        last_item_code = last_item.item_code
        last_unit_price = last_item.unit_price

    return SngInvoiceResult(
        header=header,
        line_items=line_items,
        raw_text=text[:1000] if text else "",
        last_item_code=last_item_code,
        last_unit_price=last_unit_price
    )


def extract_header(text: str) -> SngInvoiceHeader:
    """
    헤더 정보 추출 (Invoice NO, Invoice Date)

    날짜 추출 우선순위:
    1. INVOICE DATE 레이블 근처 날짜
    2. 문서 상단 첫 번째 유효 날짜
    """
    header = SngInvoiceHeader()

    # 텍스트 정규화 (날짜 내 특수문자 보정)
    normalized_text = text
    # 07/€2/2026 같은 OCR 노이즈 보정
    normalized_text = re.sub(r'(\d{1,2})/[€$@#](\d)/(\d{4})', r'\1/0\2/\3', normalized_text)

    # Invoice NO 추출 (10자리 숫자)
    invoice_no_patterns = [
        r'Invoice\s*(?:NO\.?|#)\s*(\d{10})',
        r'(\d{10})',  # 10자리 숫자
    ]

    for pattern in invoice_no_patterns:
        match = re.search(pattern, normalized_text, re.IGNORECASE)
        if match:
            header.invoice_number = match.group(1)
            break

    # Invoice Date 추출 - 우선순위 적용
    lines = normalized_text.split('\n')

    # 1순위: INVOICE DATE 레이블이 있는 줄 또는 바로 아래 줄
    for i, line in enumerate(lines[:25]):  # 상단 25줄 확인
        if 'INVOICE' in line.upper() and 'DATE' in line.upper():
            # 같은 줄에서 날짜 찾기
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', line)
            if date_match:
                header.invoice_date = date_match.group(1)
                break
            # 다음 줄에서 날짜 찾기
            if i + 1 < len(lines):
                date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', lines[i + 1])
                if date_match:
                    header.invoice_date = date_match.group(1)
                    break

    # 2순위: 상단 10줄 내 첫 번째 날짜 (INVOICE DATE 못 찾은 경우)
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
    라인 아이템 추출 - 상태 기반 파서

    SNG 인보이스 구조:
    Line 1: [PackedBy] [OrderQty] [ShipQty] [ITEM_CODE] [Desc] [ListPrice] [ListExtended]
    Line 2: [YourPrice] [YourExtended] [Discount]
    Line 3+: [Color-Qty] [Color-Qty] [Color-Qty] [Color-Qty]

    멀티 페이지 지원:
    - prev_item_code: 이전 페이지의 마지막 아이템 코드
    - prev_unit_price: 이전 페이지의 마지막 단가
    - 페이지 시작 시 아이템 코드 없이 색상만 있으면 prev_item_code에 연결

    상태 흐름:
    IDLE → ITEM_START (아이템 코드 발견, List Price 줄)
    ITEM_START → PRICE_LINE (Your Price 줄)
    PRICE_LINE → COLOR_COLLECT (색상-수량 수집)
    COLOR_COLLECT → ITEM_START (다음 아이템) 또는 계속 수집
    """
    line_items = []
    lines = text.split('\n')

    # 현재 아이템 컨텍스트 (이전 페이지에서 이어받기)
    current_item_code = prev_item_code
    current_description = None
    current_unit_price = prev_unit_price
    current_ship_qty = 0
    current_has_colors = False
    state = ParserState.COLOR_COLLECT if prev_item_code else ParserState.IDLE

    # 아이템 코드 패턴 (숫자 앞 노이즈 허용)
    item_line_pattern = r'(?:^|\s)(\d+)\s+(\d+)\s+([A-Za-z][A-Za-z0-9]{4,8})\s+([A-Z]{2})\b'

    # 색상-수량 패턴 (대시 양옆 공백 optional)
    # C-42730 - 4, P27/30 - 6 같은 복합 색상 코드 지원
    color_qty_pattern = r'([A-Z0-9][A-Z0-9/]*(?:-[A-Z0-9]+)?)\s*[-–—]\s*(\d{1,3})(?:\s*\((\d+)\))?'

    # 가격 패턴
    price_pattern = r'(\d{1,3}\.\d{2})'

    def save_item_if_no_colors():
        """색상 정보 없는 아이템 저장 (ship_qty를 quantity로 사용)"""
        nonlocal current_has_colors
        if current_item_code and not current_has_colors:
            line_items.append(SngLineItem(
                item_code=current_item_code,
                color="",  # 색상 없음
                quantity=current_ship_qty,
                unit_price=current_unit_price,
                description=current_description
            ))
            logger.info(f"색상 없는 아이템 추가: {current_item_code}, qty={current_ship_qty}")

    for i, line in enumerate(lines):
        original_line = line.strip()
        if not original_line:
            continue

        # 라인 정규화 적용
        normalized_line = normalize_line(original_line)
        line_upper = normalized_line.upper()

        # 헤더 행 제외
        if re.search(r'\b(PACKED|ORDERED|SHIPPED|DESCRIPTION|PRICE|EXTENDED|ITEM\s*NUMBER)\b', line_upper):
            continue

        # 아이템 코드 찾기
        item_match = re.search(item_line_pattern, line_upper)
        if item_match:
            order_qty = item_match.group(1)
            ship_qty = item_match.group(2)
            potential_item = item_match.group(3).upper()
            desc_prefix = item_match.group(4)

            # S로 시작하는 SNG 아이템 코드인지 확인
            if potential_item.startswith('S'):
                # 이전 아이템이 색상 없으면 저장
                save_item_if_no_colors()

                corrected_item = correct_ocr_item_code(potential_item)

                # 새 아이템 시작
                current_item_code = corrected_item
                current_unit_price = None
                current_ship_qty = int(ship_qty)
                current_has_colors = False
                state = ParserState.ITEM_START

                logger.info(f"아이템 코드 감지: {potential_item} → {current_item_code} (Qty: {order_qty}/{ship_qty}, 설명: {desc_prefix})")

                # Description 추출
                desc_match = re.search(
                    rf'{re.escape(desc_prefix)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)',
                    line_upper
                )
                if desc_match:
                    current_description = f"{desc_prefix} {desc_match.group(1).strip()}"
                else:
                    current_description = desc_prefix

                continue

        # 가격 줄 처리 (Your Price 추출)
        if state == ParserState.ITEM_START and current_item_code:
            prices = re.findall(price_pattern, normalized_line)
            if prices and len(prices) >= 1:
                current_unit_price = float(prices[0])
                logger.info(f"Your Price 추출: {current_unit_price} (아이템: {current_item_code})")
                state = ParserState.COLOR_COLLECT

        # 색상-수량 쌍 추출
        if current_item_code:
            color_qty_matches = re.findall(color_qty_pattern, line_upper)

            for match in color_qty_matches:
                color = match[0].strip()
                quantity = int(match[1])

                if is_valid_color(color):
                    line_items.append(SngLineItem(
                        item_code=current_item_code,
                        color=color,
                        quantity=quantity,
                        unit_price=current_unit_price,
                        description=current_description
                    ))
                    current_has_colors = True
                    logger.debug(f"색상-수량 추출: {current_item_code} - {color} x {quantity}")

            if color_qty_matches:
                state = ParserState.COLOR_COLLECT

    # 마지막 아이템 처리
    save_item_if_no_colors()

    logger.info(f"추출된 라인 아이템 수: {len(line_items)}")
    return line_items


def is_valid_color(color: str) -> bool:
    """
    유효한 색상 코드인지 확인
    """
    if len(color) < 1 or len(color) > 20:
        return False

    # 순수 숫자 (1, 2, 27, 530, 613 등)
    if color.isdigit():
        return True

    # 알파벳+숫자 조합 (1B, P1B/30, ASH-LATTE, COPPER 등)
    if re.match(r'^[A-Z0-9][A-Z0-9/\-]*$', color, re.IGNORECASE):
        return True

    return False


def to_dict(result: SngInvoiceResult) -> dict:
    """
    SngInvoiceResult를 JSON 직렬화 가능한 dict로 변환
    """
    return {
        "header": {
            "invoice_number": result.header.invoice_number,
            "invoice_date": result.header.invoice_date
        },
        "line_items": [
            {
                "item_code": item.item_code,
                "color": item.color,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "description": item.description
            }
            for item in result.line_items
        ],
        "raw_text_preview": result.raw_text,
        # 멀티 페이지 지원: 다음 페이지 처리 시 사용
        "last_item_code": result.last_item_code,
        "last_unit_price": result.last_unit_price
    }
