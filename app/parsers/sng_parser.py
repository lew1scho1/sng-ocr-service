"""
SNG (Shake-N-Go) 인보이스 파서

인보이스 구조:
- 헤더: Invoice NO, Invoice Date
- 라인 아이템: ITEM NUMBER, Description, 색상-수량 쌍 (4열), 단가
- 색상-수량 쌍은 여러 줄에 걸쳐 4열로 배치됨
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


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
    """
    line_items = []
    lines = text.split('\n')

    current_item_code = None
    current_description = None
    current_unit_price = None

    # ITEM NUMBER 패턴: 대문자+숫자 조합 (예: SFTWB14, SODWX24, SOHWXL3)
    item_code_pattern = r'\b([A-Z]{1,4}[A-Z0-9]{2,10})\b'

    # 색상-수량 패턴: "COLOR - QTY" 또는 "NUMBER - NUMBER"
    # 예: COPPER - 2, 1B - 4, P1B/30 - 2, ASH-LATTE - 12
    color_qty_pattern = r'([A-Z0-9][A-Z0-9/\-]*)\s*[-–—]\s*(\d{1,3})(?:\s*\((\d+)\))?'

    # 단가 패턴: 소수점 두자리 숫자 (예: 32.00, 17.00)
    price_pattern = r'\b(\d{1,3}\.\d{2})\b'

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # ITEM NUMBER 찾기 (PACKED BY, QTY 등의 헤더 행 제외)
        if re.search(r'(PACKED|ORDERED|SHIPPED|DESCRIPTION|PRICE|EXTENDED)', line, re.IGNORECASE):
            continue

        # ITEM NUMBER 추출
        item_match = re.search(item_code_pattern, line)
        if item_match:
            potential_item = item_match.group(1)
            # 유효한 ITEM CODE인지 확인 (헤더 텍스트 제외)
            if not is_header_text(potential_item):
                # 새 아이템 시작
                current_item_code = potential_item

                # Description 추출 (ITEM CODE 뒤의 텍스트)
                desc_match = re.search(rf'{re.escape(current_item_code)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)', line)
                if desc_match:
                    current_description = desc_match.group(1).strip()

                # 단가 추출
                prices = re.findall(price_pattern, line)
                if prices:
                    # Your Price는 보통 두 번째 가격 (List Price 다음)
                    current_unit_price = float(prices[-1]) if len(prices) >= 1 else None

        # 색상-수량 쌍 추출
        if current_item_code:
            color_qty_matches = re.findall(color_qty_pattern, line, re.IGNORECASE)

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
