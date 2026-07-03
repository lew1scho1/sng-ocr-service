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
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# OCR 문자 보정 맵 (자주 오인식되는 문자)
OCR_CHAR_CORRECTIONS = {
    'I': '1',  # I → 1 (아이템 코드 내 숫자)
    'O': '0',  # O → 0
    'Y': '4',  # Y → 4 (SFTWBIY → SFTWB14)
    'S': '5',  # S → 5 (숫자 위치에서만)
    'B': '8',  # B → 8 (숫자 위치에서만)
    'G': '6',  # G → 6
}


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


def correct_ocr_item_code(code: str) -> str:
    """
    OCR 오인식 문자를 보정하여 올바른 아이템 코드 반환

    예: SFTWBIY → SFTWB14
        soDwx24 → SODWX24
    """
    # 대문자로 정규화
    code = code.upper()

    # SNG 아이템 코드 구조: S + 알파벳(2-4자) + 숫자(1-3자) + 옵션
    # 예: SFTWB14, SODWX24, SOHWXL3

    # 마지막 부분이 숫자여야 하는 경우 보정
    # 보통 알파벳 뒤 숫자 부분에서 오인식 발생
    result = list(code)

    # 끝에서부터 숫자가 있어야 할 위치 찾기
    # SNG 패턴: S[A-Z]{2,4}[A-Z]?[0-9]{1,2}
    # 예: SFTWB + 14, SODWX + 24

    # 마지막 1-2자리가 숫자여야 함
    for i in range(len(result) - 1, max(len(result) - 3, 3), -1):
        char = result[i]
        if char in OCR_CHAR_CORRECTIONS and not char.isdigit():
            # 숫자로 보정
            result[i] = OCR_CHAR_CORRECTIONS[char]
            logger.debug(f"OCR 보정: {code}[{i}] '{char}' → '{result[i]}'")

    corrected = ''.join(result)
    if corrected != code:
        logger.info(f"OCR 아이템 코드 보정: {code} → {corrected}")

    return corrected


def parse_sng_invoice(text: str) -> SngInvoiceResult:
    """
    SNG 인보이스 OCR 텍스트를 파싱하여 구조화된 데이터 반환
    """
    header = extract_header(text)
    line_items = extract_line_items(text)

    return SngInvoiceResult(
        header=header,
        line_items=line_items,
        raw_text=text[:1000] if text else ""
    )


def extract_header(text: str) -> SngInvoiceHeader:
    """
    헤더 정보 추출 (Invoice NO, Invoice Date)
    """
    header = SngInvoiceHeader()

    # Invoice NO 추출 (10자리 숫자)
    # 패턴: "Invoice NO." 또는 "Invoice NO" 다음에 오는 숫자
    invoice_no_patterns = [
        r'Invoice\s*(?:NO\.?|#)\s*(\d{10})',  # Invoice NO. 3000674313
        r'(\d{10})\s*$',  # 줄 끝의 10자리 숫자
    ]

    for pattern in invoice_no_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            header.invoice_number = match.group(1)
            break

    # Invoice Date 추출 (MM/DD/YYYY 형식)
    date_patterns = [
        r'INVOICE\s*DATE\s*(\d{1,2}/\d{1,2}/\d{4})',  # INVOICE DATE 07/02/2026
        r'(\d{1,2}/\d{1,2}/\d{4})',  # 일반 날짜 패턴
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            header.invoice_date = match.group(1)
            break

    return header


def extract_line_items(text: str) -> List[SngLineItem]:
    """
    라인 아이템 추출 (ITEM NUMBER + COLOR + QUANTITY)

    SNG 인보이스 라인 구조:
    [PackedBy] [OrderQty] [ShipQty] [ITEM_CODE] [Description(2자)] [나머지...]
    예: F71    3         2         SFTWB14     HR FREETRESS WATER...

    ITEM CODE 특징:
    - 앞에 숫자(수량)가 있음
    - 뒤에 2자리 영문 설명 코드가 있음 (HR, OG 등)
    """
    line_items = []
    lines = text.split('\n')

    current_item_code = None
    current_description = None
    current_unit_price = None

    # 컨텍스트 기반 아이템 코드 패턴
    # 패턴: [숫자] [숫자] [ITEM_CODE] [영문2자]
    # 예: 3 2 SFTWB14 HR, 12 12 SODWX24 OG
    item_line_pattern = r'\d+\s+\d+\s+([A-Za-z0-9]{5,9})\s+([A-Z]{2})\s'

    # 색상-수량 패턴: "COLOR - QTY"
    # 핵심: 색상명 내부 하이픈(ASH-LATTE)은 공백 없음
    #       색상과 수량 사이 대시는 양쪽 공백 필수 (ASH-LATTE - 12)
    # 예: COPPER - 2, 1B - 4, P1B/30 - 2, ASH-LATTE - 12
    color_qty_pattern = r'([A-Z0-9][A-Z0-9/-]*[A-Z0-9]|[A-Z0-9])\s+[-–—]\s+(\d{1,3})(?:\s*\((\d+)\))?'

    # 단가 패턴: 소수점 두자리 숫자 (예: 32.00, 17.00)
    price_pattern = r'\b(\d{1,3}\.\d{2})\b'

    for i, line in enumerate(lines):
        original_line = line.strip()
        if not original_line:
            continue

        # 대문자로 정규화 (비교용)
        line_upper = original_line.upper()

        # ITEM NUMBER 찾기 (PACKED BY, QTY 등의 헤더 행 제외)
        if re.search(r'(PACKED|ORDERED|SHIPPED|DESCRIPTION|PRICE|EXTENDED)', line_upper):
            continue

        # 컨텍스트 기반 아이템 코드 추출
        # 패턴: [숫자] [숫자] [ITEM_CODE] [영문2자]
        item_match = re.search(item_line_pattern, line_upper)
        if item_match:
            potential_item = item_match.group(1).upper()
            desc_prefix = item_match.group(2)  # HR, OG 등

            # OCR 오인식 보정 적용
            corrected_item = correct_ocr_item_code(potential_item)

            # 새 아이템 시작
            current_item_code = corrected_item
            logger.info(f"아이템 코드 감지: {potential_item} → {current_item_code} (설명: {desc_prefix})")

            # Description 추출 (2자리 코드 뒤의 텍스트)
            desc_match = re.search(
                rf'{re.escape(desc_prefix)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)',
                line_upper
            )
            if desc_match:
                current_description = f"{desc_prefix} {desc_match.group(1).strip()}"
            else:
                current_description = desc_prefix

            # 단가 추출
            prices = re.findall(price_pattern, original_line)
            if prices:
                # Your Price는 보통 두 번째 가격 (List Price 다음)
                current_unit_price = float(prices[-1]) if len(prices) >= 1 else None

        # 색상-수량 쌍 추출
        if current_item_code:
            color_qty_matches = re.findall(color_qty_pattern, line_upper)

            for match in color_qty_matches:
                color = match[0].strip()
                quantity = int(match[1])
                # backorder = int(match[2]) if match[2] else 0  # 백오더 수량

                # 유효한 색상인지 확인 (숫자만 있는 것도 색상으로 인정)
                if is_valid_color(color):
                    line_items.append(SngLineItem(
                        item_code=current_item_code,
                        color=color,
                        quantity=quantity,
                        unit_price=current_unit_price,
                        description=current_description
                    ))

    logger.info(f"추출된 라인 아이템 수: {len(line_items)}")
    return line_items


def is_header_text(text: str) -> bool:
    """
    헤더/라벨 텍스트인지 확인
    """
    header_words = [
        'INVOICE', 'DATE', 'PACKED', 'ORDERED', 'SHIPPED', 'ITEM', 'NUMBER',
        'DESCRIPTION', 'PRICE', 'EXTENDED', 'AMOUNT', 'TOTAL', 'WEIGHT',
        'CUSTOMER', 'BILL', 'SHIP', 'VIA', 'GROUND', 'UPS', 'FEDEX',
        'TERMS', 'REFERENCE', 'ORDER', 'SALESPERSON', 'COPY', 'STANDARD',
        'LIST', 'YOUR', 'DISCOUNTED', 'COD', 'BOX'
    ]
    return text.upper() in header_words


def is_valid_color(color: str) -> bool:
    """
    유효한 색상 코드인지 확인
    """
    # 너무 짧거나 긴 것 제외
    if len(color) < 1 or len(color) > 20:
        return False

    # 순수 숫자도 색상으로 인정 (1, 2, 27, 530, 613 등)
    if color.isdigit():
        return True

    # 알파벳+숫자 조합 (1B, P1B/30, ASH-LATTE 등)
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
        "raw_text_preview": result.raw_text
    }
