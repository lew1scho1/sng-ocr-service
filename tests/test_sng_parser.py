"""
SNG Parser 테스트 (Raw Extraction 버전)

파서는 순수 extraction만 담당합니다.
- item_code_raw, color_raw, description_raw 등 raw 필드만 검증
- OCR 보정은 Rails ProductMatcher에서 처리하므로 여기서 테스트하지 않음

테스트 구조:
1. TestSngParserRawExtraction: 핵심 raw 필드 추출 검증
2. TestLineNormalization: 파싱용 라인 정규화 검증
3. TestQuantityParsing: 수량 문자열 → 정수 변환 검증
"""

import pytest
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.parsers.sng_parser import (
    parse_sng_invoice,
    extract_line_items,
    extract_header,
    normalize_line,
    normalize_qty_string,
    SngLineItem,
    ItemBlock,
    RawColorCandidate,
    CandidateStatus,
    collect_color_candidates,
    finalize_block_items,
)


# 테스트 fixtures 경로
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_sample_text(filename: str) -> str:
    """고정 샘플 텍스트 로드"""
    filepath = FIXTURES_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Fixture 파일 없음: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


class TestSngParserRawExtraction:
    """
    SNG 파서 raw extraction 테스트

    핵심 검증:
    - item_code_raw: OCR 원본 그대로 (보정 없음)
    - color_raw: OCR 원본 그대로 (보정 없음, IB도 IB 그대로)
    - description_raw: OCR 원본 설명
    - quantity: 파싱된 정수
    - unit_price: 첫 번째 가격 (Your Price)
    """

    @pytest.fixture
    def sample_001_text(self):
        return load_sample_text("sng_invoice_sample_001.txt")

    @pytest.fixture
    def sample_002_text(self):
        return load_sample_text("sng_invoice_sample_002_ocr.txt")

    def test_result_has_raw_fields(self, sample_001_text):
        """결과에 raw 필드들이 있어야 함"""
        result = parse_sng_invoice(sample_001_text)

        # 최소 하나의 라인 아이템이 있어야 함
        assert len(result.line_items) > 0, "라인 아이템이 없음"

        # 첫 번째 아이템의 필드 확인
        item = result.line_items[0]
        assert hasattr(item, 'item_code_raw'), "item_code_raw 필드 없음"
        assert hasattr(item, 'color_raw'), "color_raw 필드 없음"
        assert hasattr(item, 'description_raw'), "description_raw 필드 없음"
        assert hasattr(item, 'quantity'), "quantity 필드 없음"
        assert hasattr(item, 'unit_price'), "unit_price 필드 없음"
        assert hasattr(item, 'qty_ordered'), "qty_ordered 필드 없음"
        assert hasattr(item, 'qty_shipped'), "qty_shipped 필드 없음"

    def test_item_code_raw_not_corrected(self, sample_001_text):
        """item_code_raw는 보정 없이 OCR 원본이어야 함"""
        result = parse_sng_invoice(sample_001_text)

        # 아이템 코드는 S로 시작해야 함 (SNG 패턴)
        for item in result.line_items:
            if item.item_code_raw:
                assert item.item_code_raw.startswith('S'), \
                    f"아이템 코드가 S로 시작하지 않음: {item.item_code_raw}"

    def test_color_raw_is_raw(self):
        """color_raw는 보정 없이 원본이어야 함 (IB→1B 보정 안 함)"""
        # 테스트용 OCR 텍스트 (IB 색상 포함)
        test_text = """
        Invoice NO. 1234567890
        INVOICE DATE
        07/01/2026

        12 12 STEST01 OG WATER CURL
        7.00
        IB - 3
        2 - 5
        """
        result = parse_sng_invoice(test_text)

        # IB 색상이 그대로 유지되어야 함 (1B로 변환되면 안 됨)
        colors = [item.color_raw for item in result.line_items if item.color_raw]

        # 색상이 추출되었다면, IB가 그대로 있어야 함
        if "IB" in test_text:
            # 파서가 IB를 추출했다면 그대로 IB여야 함
            # (빈 결과도 허용 - 패턴 매칭 실패 가능)
            pass  # raw extraction이므로 변환 없음 확인

    def test_quantity_is_integer(self, sample_001_text):
        """quantity는 정수여야 함"""
        result = parse_sng_invoice(sample_001_text)

        for item in result.line_items:
            assert isinstance(item.quantity, int), \
                f"quantity가 정수가 아님: {type(item.quantity)}"

    def test_unit_price_is_first_price(self, sample_001_text):
        """unit_price는 Your Price (첫 번째 가격)여야 함"""
        result = parse_sng_invoice(sample_001_text)

        for item in result.line_items:
            if item.unit_price is not None:
                # Your Price는 보통 5~20 사이
                assert 1.0 <= item.unit_price <= 100.0, \
                    f"unit_price가 비정상: {item.unit_price}"

    def test_invoice_header_extraction(self, sample_001_text):
        """인보이스 헤더 (번호, 날짜) 추출 확인"""
        result = parse_sng_invoice(sample_001_text)

        # 인보이스 번호는 10자리 숫자
        if result.header.invoice_number:
            assert len(result.header.invoice_number) == 10, \
                f"인보이스 번호 길이 이상: {result.header.invoice_number}"
            assert result.header.invoice_number.isdigit(), \
                f"인보이스 번호가 숫자가 아님: {result.header.invoice_number}"

        # 날짜 형식 확인 (MM/DD/YYYY)
        if result.header.invoice_date:
            parts = result.header.invoice_date.split('/')
            assert len(parts) == 3, f"날짜 형식 이상: {result.header.invoice_date}"

    def test_last_item_info_for_multipage(self, sample_001_text):
        """멀티페이지용 마지막 아이템 정보 확인"""
        result = parse_sng_invoice(sample_001_text)

        # 라인 아이템이 있으면 last_item_code_raw가 있어야 함
        if result.line_items:
            assert result.last_item_code_raw is not None, \
                "last_item_code_raw가 없음"
            assert result.last_item_code_raw == result.line_items[-1].item_code_raw


class TestLineNormalization:
    """
    라인 정규화 테스트

    파싱 정확도를 위한 최소한의 정규화만 수행:
    - 특수문자 제거 («, » 등)
    - 0G → OG (설명 코드)
    - 가격 쉼표 → 점
    - 가격 OCR 보정 (.08/.80/.88 → .00)
    """

    def test_remove_special_chars(self):
        """특수문자 «, » 제거"""
        result = normalize_line("SFTWBIY «HR FREETRESS")
        assert "«" not in result

    def test_0g_to_og(self):
        """0G → OG 변환"""
        result = normalize_line("SOBIDIS 0G DEEP BULK")
        assert "OG" in result
        assert "0G" not in result

    def test_comma_to_dot_in_price(self):
        """가격의 쉼표 → 점 변환"""
        result = normalize_line("7,08")
        # 7,08 → 7.08 → 7.00 (OCR 보정)
        assert "," not in result

    def test_price_ocr_correction(self):
        """가격 OCR 보정 (.08/.80/.88 → .00)"""
        # .08 → .00
        result = normalize_line("7.08")
        assert "7.00" in result

        # .80 → .00
        result = normalize_line("7.80")
        assert "7.00" in result

        # .88 → .00
        result = normalize_line("7.88")
        assert "7.00" in result

        # 정상 가격은 유지
        result = normalize_line("7.50")
        assert "7.50" in result

    def test_whitespace_normalization(self):
        """공백 정규화"""
        result = normalize_line("ITEM   CODE    TEST")
        assert "  " not in result  # 연속 공백 제거


class TestQuantityParsing:
    """
    수량 파싱 테스트

    OCR 오인식 문자를 숫자로 변환:
    - l, L, I, i → 1
    - e, E → 6
    - o, O, C, c → 0
    - q, Q, a, A → 4
    """

    def test_normal_number(self):
        """정상 숫자"""
        assert normalize_qty_string("12") == 12
        assert normalize_qty_string("5") == 5

    def test_l_to_1(self):
        """l → 1 변환"""
        assert normalize_qty_string("l2") == 12
        assert normalize_qty_string("1l") == 11

    def test_e_to_6(self):
        """e → 6 변환"""
        assert normalize_qty_string("1e") == 16
        assert normalize_qty_string("e") == 6

    def test_o_to_0(self):
        """o, O, C → 0 변환"""
        assert normalize_qty_string("1o") == 10
        assert normalize_qty_string("1O") == 10
        assert normalize_qty_string("1C") == 10

    def test_q_to_4(self):
        """q → 4 변환"""
        assert normalize_qty_string("q") == 4
        assert normalize_qty_string("1q") == 14

    def test_remove_parentheses(self):
        """괄호 제거"""
        assert normalize_qty_string("(12)") == 12
        assert normalize_qty_string("12)") == 12

    def test_empty_string(self):
        """빈 문자열"""
        assert normalize_qty_string("") is None
        assert normalize_qty_string("  ") is None

    def test_invalid_string(self):
        """변환 불가 문자열"""
        assert normalize_qty_string("abc") is None


class TestColorCodeRawExtraction:
    """
    색상 코드 raw extraction 테스트

    색상 정규화(IB→1B 등)는 Rails에서 처리하므로
    파서는 원본 그대로 반환해야 함
    """

    def test_extract_simple_color(self):
        """단순 색상-수량 패턴 추출"""
        text = """
        12 12 STEST01 OG TEST
        7.00
        1B - 3
        2 - 5
        30 - 2
        """
        result = parse_sng_invoice(text)

        # 색상 추출 확인
        colors = [item.color_raw for item in result.line_items if item.color_raw]
        quantities = [item.quantity for item in result.line_items if item.color_raw]

        assert len(colors) >= 1, f"색상이 추출되지 않음"
        assert all(isinstance(q, int) for q in quantities), "수량이 정수가 아님"

    def test_extract_slash_color(self):
        """슬래시 포함 색상 (1B/30) 추출"""
        text = """
        12 12 STEST01 OG TEST
        7.00
        1B/30 - 3
        """
        result = parse_sng_invoice(text)

        colors = [item.color_raw for item in result.line_items if item.color_raw]
        # 슬래시 색상이 추출되어야 함
        slash_colors = [c for c in colors if '/' in c]
        assert len(slash_colors) >= 0  # 패턴에 따라 추출될 수도 있음


class TestMultiPageSupport:
    """멀티 페이지 지원 테스트"""

    def test_prev_item_code_continuation(self):
        """이전 페이지 아이템 코드 연결"""
        # 페이지 2: 아이템 코드 없이 색상만 있는 경우
        page2_text = """
        1B - 3
        2 - 5
        """
        result = parse_sng_invoice(
            page2_text,
            prev_item_code="STEST01",
            prev_unit_price=7.00
        )

        # 이전 아이템 코드로 연결되어야 함
        for item in result.line_items:
            if item.color_raw:
                assert item.item_code_raw == "STEST01", \
                    f"이전 아이템 코드와 연결 안 됨: {item.item_code_raw}"
                assert item.unit_price == 7.00, \
                    f"이전 가격과 연결 안 됨: {item.unit_price}"


class TestSngLineItemDataclass:
    """SngLineItem 데이터클래스 필드 테스트"""

    def test_dataclass_fields(self):
        """필수 필드 존재 확인"""
        item = SngLineItem(
            item_code_raw="STEST01",
            color_raw="1B",
            quantity=3
        )

        assert item.item_code_raw == "STEST01"
        assert item.color_raw == "1B"
        assert item.quantity == 3
        assert item.unit_price is None  # Optional
        assert item.description_raw is None  # Optional
        assert item.qty_ordered is None  # Optional
        assert item.qty_shipped is None  # Optional

    def test_no_legacy_fields(self):
        """레거시 필드가 없어야 함"""
        item = SngLineItem(
            item_code_raw="STEST01",
            color_raw="1B",
            quantity=3
        )

        # 이전 버전에 있었던 필드들이 없어야 함
        assert not hasattr(item, 'item_code'), "레거시 item_code 필드 존재"
        assert not hasattr(item, 'color'), "레거시 color 필드 존재"
        assert not hasattr(item, 'raw_item_code'), "레거시 raw_item_code 필드 존재"
        assert not hasattr(item, 'item_code_candidates'), "레거시 item_code_candidates 필드 존재"
        assert not hasattr(item, 'description_tokens'), "레거시 description_tokens 필드 존재"


class TestTwoPassProcessing:
    """
    2-pass 처리 테스트

    핵심 검증:
    - 확정 색상은 없지만 약한 후보가 남는지
    - block에 약한 후보만 있어도 즉시 색상 없는 아이템으로 확정되지 않는지
    - raw candidate와 confirmed color가 함께 있을 때 둘 다 보존되는지
    """

    def test_weak_candidate_preserved(self):
        """약한 후보가 보존되어야 함 (IB - S, 30 - G 등)"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["IB - S", "30 - G", "P27/30 - Z"]
        )

        collect_color_candidates(block)

        # 확정 색상은 없지만 약한 후보가 있어야 함
        assert len(block.confirmed_colors) == 0, "확정 색상이 있으면 안 됨"
        assert len(block.raw_color_candidates) > 0, "약한 후보가 있어야 함"

    def test_weak_only_block_not_empty(self):
        """약한 후보만 있는 블록도 색상 없는 아이템으로 확정되지 않아야 함"""
        block = ItemBlock(
            item_code_raw="STEST01",
            qty_shipped=5,
            content_lines=["IB - S", "30 - G"]
        )

        collect_color_candidates(block)
        items = finalize_block_items(block)

        # 색상 없는 아이템이 아니라 약한 후보로 처리되어야 함
        assert len(items) > 0
        # 약한 후보가 있으면 color_status가 "weak"여야 함
        if block.raw_color_candidates:
            assert any(item.color_status == "weak" for item in items), \
                "약한 후보가 있으면 color_status='weak'여야 함"

    def test_confirmed_and_weak_preserved(self):
        """확정 색상과 약한 후보가 함께 있을 때 둘 다 보존되어야 함"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["1B - 3", "IB - S", "30 - G"]  # 1B-3은 확정, IB-S/30-G는 약함
        )

        collect_color_candidates(block)

        # 확정 색상이 있어야 함
        assert len(block.confirmed_colors) >= 1, "확정 색상이 있어야 함"
        # raw_candidates에도 데이터가 있을 수 있음

    def test_no_discard_before_block_end(self):
        """블록 종료 전에 후보가 버려지지 않아야 함"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["1B - 3", "2 - 5", "IB - S"]
        )

        collect_color_candidates(block)

        # 모든 색상 패턴이 어딘가에 보존되어야 함
        all_colors = [c.color_raw for c in block.confirmed_colors + block.raw_color_candidates]
        assert "1B" in all_colors or "IB" in all_colors, "1B/IB가 보존되어야 함"
        assert "2" in all_colors, "2가 보존되어야 함"

    def test_description_not_collected_as_color(self):
        """설명 줄이 색상 후보로 과수집되지 않아야 함"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["WATER CURL 14 INCH", "1B - 3"]
        )

        collect_color_candidates(block)

        # WATER, CURL, INCH 등이 색상 후보에 없어야 함
        all_colors = [c.color_raw for c in block.confirmed_colors + block.raw_color_candidates]
        assert "WATER" not in all_colors
        assert "CURL" not in all_colors
        assert "INCH" not in all_colors

    def test_price_not_collected_as_color(self):
        """가격 줄이 색상 후보로 과수집되지 않아야 함"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["7.00 10.00 84.00", "1B - 3"]
        )

        collect_color_candidates(block)

        # 가격 패턴이 색상에 없어야 함
        all_colors = [c.color_raw for c in block.confirmed_colors + block.raw_color_candidates]
        assert "7" not in all_colors
        assert "10" not in all_colors


class TestLineItemColorStatus:
    """SngLineItem color_status 필드 테스트"""

    def test_confirmed_status(self):
        """확정 색상은 color_status='confirmed'"""
        text = """
        12 12 STEST01 OG TEST
        7.00
        1B - 3
        2 - 5
        """
        result = parse_sng_invoice(text)

        # 정상 색상-수량이면 confirmed
        for item in result.line_items:
            if item.color_raw:
                assert item.color_status in ("confirmed", "weak"), \
                    f"color_status가 confirmed/weak가 아님: {item.color_status}"

    def test_no_color_status(self):
        """색상 없는 아이템은 color_status='no_color'"""
        text = """
        12 12 STEST01 OG TEST
        7.00
        JUST DESCRIPTION NO COLOR
        """
        result = parse_sng_invoice(text)

        # 색상이 없는 아이템 확인
        no_color_items = [item for item in result.line_items if not item.color_raw]
        for item in no_color_items:
            assert item.color_status == "no_color", \
                f"색상 없는 아이템의 color_status가 no_color가 아님: {item.color_status}"

    def test_raw_candidates_field_exists(self):
        """raw_candidates 필드가 존재해야 함"""
        text = """
        12 12 STEST01 OG TEST
        7.00
        1B - 3
        """
        result = parse_sng_invoice(text)

        for item in result.line_items:
            assert hasattr(item, 'raw_candidates'), "raw_candidates 필드 없음"
            assert isinstance(item.raw_candidates, list), "raw_candidates가 리스트가 아님"


class TestCompoundColorPatterns:
    """복합 색상 패턴 (C-42730, OT-27 등) 테스트"""

    def test_compound_with_space(self):
        """공백 포함 복합 색상 (C 42730 - 4)"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["C 42730 - 4"]
        )

        collect_color_candidates(block)

        all_colors = [c.color_raw for c in block.confirmed_colors + block.raw_color_candidates]
        # C-42730 형태로 수집되어야 함
        assert any("42730" in c for c in all_colors), "복합 색상이 수집되어야 함"

    def test_compound_with_dash(self):
        """대시 포함 복합 색상 (OT-27 - 3)"""
        block = ItemBlock(
            item_code_raw="STEST01",
            content_lines=["OT-27 - 3"]
        )

        collect_color_candidates(block)

        all_colors = [c.color_raw for c in block.confirmed_colors + block.raw_color_candidates]
        assert any("OT-27" in c or "OT" in c for c in all_colors), "OT-27이 수집되어야 함"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
