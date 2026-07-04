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

# OCR 오인식 매핑 테이블
QTY_OCR_MAP = {
    'l': '1', 'L': '1', 'I': '1', 'i': '1',
    'e': '6', 'E': '6',
    'o': '0', 'O': '0', 'C': '0', 'c': '0',
    'q': '4', 'Q': '4',
    'a': '4', 'A': '4',
    ')': '', '(': '', ' ': ''
}

# 아이템 코드 OCR 보정 - 숫자 위치 (끝 2자리만)
# 보수적 접근: 명확한 케이스만 보정
# L, B, A는 문자일 가능성이 높아 제외
DIGIT_POSITION_MAP = {
    'I': '1',             # I→1 (SOCRDI2 → SOCRD12)
    'O': '0', 'o': '0',   # O→0
    'S': '8', 's': '8',   # S→8 (SOATXIS → SOATX18)
    'Y': '4', 'y': '4',   # Y→4 (SOATXIY → SOATX14)
    # 'L': '1',           # L은 SOHWXL3처럼 문자일 수 있어 제외
    # 'B': '8',           # B는 SFTWB14처럼 문자일 수 있어 제외
    # 'A': '4',           # A는 SOB4A12처럼 문자일 수 있어 제외
    # 'Z': '2',           # Z는 드물어서 제외
}

# 아이템 코드 OCR 보정 - 문자 위치 (앞 4-5자리)
LETTER_POSITION_MAP = {
    '0': 'O',
    '1': 'I',
    '8': 'B',
    '4': 'A',
}

# 특수 문자 보정 - 제거됨
# H→M은 컨텍스트 의존적이라 일괄 적용 불가
# SOAWXH3 → SOAWXM3 (O) vs SOHWXL3 → SOHWXL3 (X)
SPECIAL_CHAR_MAP = {}


# Description 토큰화용 타입 패턴
DESCRIPTION_TYPES = [
    'WATER CURL', 'WATER WAVE', 'DEEP WAVE', 'OCEAN DEEP WAVE',
    'BODY WAVE', 'BOHEMIAN CURL', 'HAWAIIAN CURL',
    'DEEP BULK', 'AFRO KINKY BULK',
    'STRAIGHT', 'ROD SET',
]

# Description 토큰화용 스타일 패턴
DESCRIPTION_STYLES = [
    'CLIP-IN', 'BULK', 'ORGANIQUE', 'FREETRESS',
]


@dataclass
class DescriptionTokens:
    """Description 토큰화 결과"""
    type: Optional[str] = None       # WATER CURL, DEEP BULK 등
    length: Optional[str] = None     # 14", 18", 24" 등
    pcs: Optional[str] = None        # 3PCS, 9PCS 등
    style: Optional[str] = None      # BULK, CLIP-IN, ORGANIQUE 등
    raw: str = ""                    # 원본 description


def tokenize_description(raw_desc: str) -> DescriptionTokens:
    """
    Description 문자열을 토큰화하여 구조화된 정보 추출

    SNG 인보이스 Description 구조:
    - [OG/HR] [TYPE] [LENGTH"] [STYLE] [PCS]
    - 예: "OG WATER CURL ORGANIQUE 14"" → type=WATER CURL, length=14", style=ORGANIQUE
    - 예: "OG DEEP BULK 18" ORGANIQUE" → type=DEEP BULK, length=18", style=ORGANIQUE
    - 예: "OG BODY WAVE 3PCS (14"16"18")" → type=BODY WAVE, pcs=3PCS

    Returns:
        DescriptionTokens: 토큰화된 Description 정보
    """
    if not raw_desc:
        return DescriptionTokens(raw="")

    desc_upper = raw_desc.upper().strip()
    tokens = DescriptionTokens(raw=raw_desc)

    # 1. Type 추출 (긴 패턴 우선)
    for type_pattern in sorted(DESCRIPTION_TYPES, key=len, reverse=True):
        if type_pattern in desc_upper:
            tokens.type = type_pattern
            break

    # 2. Length 추출 (14", 18", 22", 24" 등)
    length_match = re.search(r'\b(\d{1,2})["\'″]?\b', desc_upper)
    if length_match:
        tokens.length = f'{length_match.group(1)}"'

    # 3. PCS 추출 (3PCS, 9PCS 등)
    pcs_match = re.search(r'\b(\d+)\s*PCS?\b', desc_upper)
    if pcs_match:
        tokens.pcs = f'{pcs_match.group(1)}PCS'

    # 4. Style 추출
    for style in DESCRIPTION_STYLES:
        if style in desc_upper:
            tokens.style = style
            break

    logger.debug(f"Description 토큰화: '{raw_desc}' → type={tokens.type}, length={tokens.length}, pcs={tokens.pcs}, style={tokens.style}")

    return tokens


def generate_item_code_candidates(raw_code: str, tokens: DescriptionTokens) -> List[str]:
    """
    OCR 아이템 코드와 Description 토큰을 기반으로 가능한 SKU 후보 생성

    Args:
        raw_code: OCR로 읽은 원본 아이템 코드 (예: SOATXIY, SOBIDIS)
        tokens: Description 토큰화 결과

    Returns:
        List[str]: 가능한 SKU 후보 리스트 (우선순위 순)

    예:
        raw_code="SOATXIY", tokens.length="14""
        → ["SOATX14", "SOATX18", "SOATXIY", ...]
    """
    candidates = []
    code_upper = raw_code.upper()

    # 1. 기본 OCR 보정 적용
    corrected = correct_ocr_item_code(code_upper)
    if corrected not in candidates:
        candidates.append(corrected)

    # 2. Description length 기반 후보 생성
    if tokens.length:
        length_num = re.search(r'(\d+)', tokens.length)
        if length_num:
            length_str = length_num.group(1)
            # 끝 2자리를 length로 대체
            if len(code_upper) >= 2:
                length_candidate = code_upper[:-2] + length_str
                length_candidate = correct_ocr_item_code(length_candidate)
                if length_candidate not in candidates:
                    candidates.append(length_candidate)

    # 3. PCS 기반 후보 (예: 3PCS → XM3, XL3, XS3)
    if tokens.pcs:
        pcs_match = re.search(r'(\d+)', tokens.pcs)
        if pcs_match:
            pcs_num = pcs_match.group(1)
            # 끝 1자리를 PCS 숫자로 대체
            if len(code_upper) >= 1:
                pcs_candidate = code_upper[:-1] + pcs_num
                pcs_candidate = correct_ocr_item_code(pcs_candidate)
                if pcs_candidate not in candidates:
                    candidates.append(pcs_candidate)

    # 4. 원본 코드 (보정 없이)
    if code_upper not in candidates:
        candidates.append(code_upper)

    logger.debug(f"SKU 후보 생성: {raw_code} → {candidates}")

    return candidates


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
    # 확장 필드: ITEMCODE + DESCRIPTION 기반 SKU 매칭용
    raw_item_code: Optional[str] = None       # OCR 원본 아이템 코드
    item_code_candidates: Optional[List[str]] = None  # 가능한 SKU 후보 리스트
    description_tokens: Optional[DescriptionTokens] = None  # 토큰화된 Description


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


def normalize_qty_string(s: str) -> Optional[int]:
    """
    OCR 오인식된 수량 문자열을 정수로 변환

    예: 'le' → 16, 'q' → 4, 'C)' → 0, '12' → 12
    """
    if not s or not s.strip():
        return None

    cleaned = ''.join(QTY_OCR_MAP.get(c, c) for c in s.strip())

    # 숫자만 남았는지 확인
    if not cleaned:
        return 0

    try:
        return int(cleaned)
    except ValueError:
        logger.debug(f"수량 변환 실패: '{s}' → '{cleaned}'")
        return None


def correct_ocr_item_code(code: str) -> str:
    """
    OCR 오인식 문자를 보정하여 올바른 아이템 코드 반환

    SNG 코드 구조 분석:
    - 총 7-8자리: S + 문자(4-5) + 숫자(2-3)
    - 예: SOB1D18, SOATX24, SOCSX20

    보정 전략:
    1. 특수 문자 보정 (H→M 등)
    2. 끝 3자리는 숫자 위치로 가정 → 문자를 숫자로
    3. 앞 4-5자리는 문자 위치로 가정 → 숫자를 문자로
    """
    code = code.upper()
    result = list(code)
    n = len(result)

    # 1. 특수 문자 보정 (위치 무관)
    for i, char in enumerate(result):
        if char in SPECIAL_CHAR_MAP:
            result[i] = SPECIAL_CHAR_MAP[char]

    # 2. 끝 2자리만: 숫자 위치 → 문자를 숫자로
    # SOB1D18에서 '18' 부분, SOATX24에서 '24' 부분
    # B, A 등은 문자일 가능성이 있어 3자리까지 확장하지 않음
    for i in range(n - 1, max(n - 3, 2), -1):
        char = result[i]
        if char in DIGIT_POSITION_MAP:
            result[i] = DIGIT_POSITION_MAP[char]

    # 3. 중간 영역 (3~5번째): 1과 I, 0과 O 등 컨텍스트 기반 판단
    # SOB1D18의 '1D' 부분 - 숫자와 문자 혼합
    # 여기서는 연속된 숫자/문자 패턴으로 판단
    # D 앞의 1은 그대로 유지 (1D는 제품 라인)

    corrected = ''.join(result)
    if corrected != code:
        logger.info(f"OCR 아이템 코드 보정: {code} → {corrected}")

    return corrected


def normalize_color_code(color: str) -> str:
    """
    색상 코드 OCR 보정

    예: IB → 1B, I → 1
    """
    color = color.upper().strip()

    # IB → 1B (첫 글자가 I이고 두 번째가 B인 경우)
    if color.startswith('I') and len(color) >= 2 and color[1] == 'B':
        color = '1' + color[1:]
    # 단독 I → 1
    elif color == 'I':
        color = '1'

    return color


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
    라인 아이템 추출 - 상태 기반 파서 (개선판)

    SNG 인보이스 구조:
    Line 1: [PackedBy] [OrderQty] [ShipQty] [ITEM_CODE] [Desc] [ListPrice] [ListExtended]
    Line 2: [YourPrice] [YourExtended] [Discount]
    Line 3+: [Color-Qty] [Color-Qty] [Color-Qty] [Color-Qty]

    개선사항:
    - 수량 컬럼 OCR 오인식 허용 (le→16, q→4, C)→0)
    - 아이템 코드 + OG/HR 설명을 주 앵커로 사용
    - 색상 코드 OCR 보정 (IB→1B)
    """
    line_items = []
    lines = text.split('\n')

    # 현재 아이템 컨텍스트 (이전 페이지에서 이어받기)
    current_item_code = prev_item_code
    current_raw_item_code = None  # OCR 원본 코드
    current_description = None
    current_description_tokens = None  # Description 토큰
    current_item_candidates = None  # SKU 후보 리스트
    current_unit_price = prev_unit_price
    current_ship_qty = 0
    current_has_colors = False
    state = ParserState.COLOR_COLLECT if prev_item_code else ParserState.IDLE

    # 아이템 코드 패턴 - 수량 컬럼에 OCR 오인식 문자 허용
    # 수량은 숫자 또는 오인식 문자 (l, e, o, q, a, C, ) 등)
    qty_char_class = r'[0-9leoqaciLEOQACIO\)\(]+'
    # 전체 패턴: [PackedBy] [Qty1] [Qty2] [S코드] [OG/HR]
    item_line_pattern = rf'(?:^|[A-Z]{{2,3}}\s+)({qty_char_class})\s+({qty_char_class})\s+(S[A-Za-z0-9]{{4,8}})\s+(OG|HR|0G)\b'

    # 대체 패턴: S코드 + OG 직접 탐색 (수량 없어도 감지)
    item_code_only_pattern = r'\b(S[A-Za-z0-9]{5,8})\s+(OG|HR|0G)\s+(.+?)\s+(\d{1,3}\.\d{2})'

    # 색상-수량 패턴
    # 1B - 6, 30 - 6, C-42730 - 4, P27/30 - 6 등
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
                description=current_description,
                raw_item_code=current_raw_item_code,
                item_code_candidates=current_item_candidates,
                description_tokens=current_description_tokens
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

        # 아이템 코드 찾기 - 1차: 수량 포함 패턴
        item_match = re.search(item_line_pattern, line_upper)
        if item_match:
            qty1_raw = item_match.group(1)
            qty2_raw = item_match.group(2)
            potential_item = item_match.group(3).upper()
            desc_prefix = item_match.group(4).upper()

            # 수량 OCR 보정
            order_qty = normalize_qty_string(qty1_raw)
            ship_qty = normalize_qty_string(qty2_raw)

            if order_qty is None:
                order_qty = 0
            if ship_qty is None:
                ship_qty = 0

            # S로 시작하는 SNG 아이템 코드
            if potential_item.startswith('S'):
                # 이전 아이템이 색상 없으면 저장
                save_item_if_no_colors()

                # OCR 원본 코드 저장
                current_raw_item_code = potential_item

                # Description 추출 (토큰화를 위해 먼저 추출)
                desc_match = re.search(
                    rf'{re.escape(desc_prefix)}\s+(.+?)(?:\d{{1,3}}\.\d{{2}}|$)',
                    line_upper
                )
                if desc_match:
                    current_description = f"{desc_prefix} {desc_match.group(1).strip()}"
                else:
                    current_description = desc_prefix

                # Description 토큰화
                current_description_tokens = tokenize_description(current_description)

                # Description 토큰 기반 SKU 후보 생성
                current_item_candidates = generate_item_code_candidates(
                    potential_item, current_description_tokens
                )

                # 첫 번째 후보를 최종 아이템 코드로 사용 (OCR 보정 + Description 기반)
                corrected_item = current_item_candidates[0] if current_item_candidates else correct_ocr_item_code(potential_item)

                # 새 아이템 시작
                current_item_code = corrected_item
                current_unit_price = None
                current_ship_qty = ship_qty
                current_has_colors = False
                state = ParserState.ITEM_START

                logger.info(f"아이템 코드 감지: {potential_item} → {current_item_code} (Qty: {order_qty}/{ship_qty}, 설명: {desc_prefix}, 후보: {current_item_candidates})")

                continue

        # 2차: 대체 패턴 - 수량 없이 S코드 + OG 직접 탐색
        if not item_match:
            alt_match = re.search(item_code_only_pattern, line_upper)
            if alt_match:
                potential_item = alt_match.group(1).upper()
                desc_prefix = alt_match.group(2).upper()
                desc_rest = alt_match.group(3).strip()
                first_price = alt_match.group(4)

                if potential_item.startswith('S'):
                    # 이전 아이템이 색상 없으면 저장
                    save_item_if_no_colors()

                    # OCR 원본 코드 저장
                    current_raw_item_code = potential_item
                    current_description = f"{desc_prefix} {desc_rest}"

                    # Description 토큰화
                    current_description_tokens = tokenize_description(current_description)

                    # Description 토큰 기반 SKU 후보 생성
                    current_item_candidates = generate_item_code_candidates(
                        potential_item, current_description_tokens
                    )

                    # 첫 번째 후보를 최종 아이템 코드로 사용
                    corrected_item = current_item_candidates[0] if current_item_candidates else correct_ocr_item_code(potential_item)

                    # 새 아이템 시작
                    current_item_code = corrected_item
                    current_unit_price = None
                    current_ship_qty = 0  # 수량 불명
                    current_has_colors = False
                    state = ParserState.ITEM_START

                    logger.info(f"아이템 코드 감지 (대체패턴): {potential_item} → {current_item_code} (후보: {current_item_candidates})")
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
                raw_color = match[0].strip()
                quantity = int(match[1])

                # 색상 코드 OCR 보정 (IB → 1B)
                color = normalize_color_code(raw_color)

                if is_valid_color(color):
                    line_items.append(SngLineItem(
                        item_code=current_item_code,
                        color=color,
                        quantity=quantity,
                        unit_price=current_unit_price,
                        description=current_description,
                        raw_item_code=current_raw_item_code,
                        item_code_candidates=current_item_candidates,
                        description_tokens=current_description_tokens
                    ))
                    current_has_colors = True
                    if raw_color != color:
                        logger.debug(f"색상-수량 추출: {current_item_code} - {raw_color}→{color} x {quantity}")
                    else:
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
    def tokens_to_dict(tokens: Optional[DescriptionTokens]) -> Optional[dict]:
        if tokens is None:
            return None
        return {
            "type": tokens.type,
            "length": tokens.length,
            "pcs": tokens.pcs,
            "style": tokens.style,
            "raw": tokens.raw
        }

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
                "description": item.description,
                # ITEMCODE + DESCRIPTION 기반 SKU 매칭용 확장 필드
                "raw_item_code": item.raw_item_code,
                "item_code_candidates": item.item_code_candidates,
                "description_tokens": tokens_to_dict(item.description_tokens)
            }
            for item in result.line_items
        ],
        "raw_text_preview": result.raw_text,
        # 멀티 페이지 지원: 다음 페이지 처리 시 사용
        "last_item_code": result.last_item_code,
        "last_unit_price": result.last_unit_price
    }
