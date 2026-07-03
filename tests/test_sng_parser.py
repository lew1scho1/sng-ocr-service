"""
SNG Parser 테스트

고정 샘플 OCR 텍스트로 파서 로직만 검증합니다.
OCR 품질 문제는 별도로 처리합니다.
"""

import pytest
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.parsers.sng_parser import parse_sng_invoice, extract_line_items, extract_header


# 테스트 fixtures 경로
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_sample_text(filename: str) -> str:
    """고정 샘플 텍스트 로드"""
    filepath = FIXTURES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


class TestSngParserWithRealOCR:
    """실제 OCR 결과로 파서 테스트"""

    @pytest.fixture
    def sample_001_text(self):
        return load_sample_text("sng_invoice_sample_001.txt")

    def test_should_detect_5_item_codes(self, sample_001_text):
        """5개 아이템 코드가 감지되어야 함"""
        result = parse_sng_invoice(sample_001_text)

        # 현재 파서는 2개만 감지 (SODWX24, SOHWXM3)
        # 수정 후 5개 감지 기대: SFTWB14, SODWX24, SOHWXL3, SOHWXL3, SOHWXM3
        item_codes = [item.item_code for item in result.line_items]
        unique_codes = list(dict.fromkeys(item_codes))  # 순서 유지하면서 중복 제거

        # 기대하는 아이템 코드들
        expected_codes = ["SFTWB14", "SODWX24", "SOHWXL3", "SOHWXM3"]

        for code in expected_codes:
            assert code in unique_codes, f"{code}가 감지되지 않음. 감지된 코드: {unique_codes}"

    def test_should_extract_invoice_date_correctly(self, sample_001_text):
        """인보이스 날짜가 07/02/2026이어야 함 (08/31/2026 아님)"""
        result = parse_sng_invoice(sample_001_text)

        # OCR 원문에서 07/€2/2026로 읽혔지만, 정규화 후 07/02/2026이 되어야 함
        # 또는 INVOICE DATE 근처의 첫 번째 날짜여야 함
        assert result.header.invoice_date == "07/02/2026", \
            f"날짜가 잘못됨: {result.header.invoice_date}"

    def test_should_extract_invoice_number(self, sample_001_text):
        """인보이스 번호가 3000674313이어야 함"""
        result = parse_sng_invoice(sample_001_text)

        assert result.header.invoice_number == "3000674313", \
            f"인보이스 번호가 잘못됨: {result.header.invoice_number}"

    def test_should_extract_color_qty_without_spaces(self, sample_001_text):
        """공백 없는 색상-수량 패턴(1-4, 27-2)도 추출해야 함"""
        result = parse_sng_invoice(sample_001_text)

        # SOHWXL3 아이템에서 색상-수량 쌍 추출 확인
        sohwxl3_items = [item for item in result.line_items if item.item_code == "SOHWXL3"]

        # 최소 8개의 색상-수량 쌍이 있어야 함 (1-4, 1-4, 2-3, 4-2, 27-2, 30-2, 530-2, 613-2)
        assert len(sohwxl3_items) >= 8, \
            f"SOHWXL3 색상-수량 쌍이 부족함: {len(sohwxl3_items)}개"

    def test_should_extract_correct_unit_price(self, sample_001_text):
        """unit_price는 Your Price여야 함 (List Price나 Extended가 아님)"""
        result = parse_sng_invoice(sample_001_text)

        # SODWX24의 unit_price 확인
        # 원본: List=10.00, Your=7.00, List Extended=120.00, Your Extended=84.00
        # OCR: 7.08 (OCR 오차)
        sodwx24_items = [item for item in result.line_items if item.item_code == "SODWX24"]

        if sodwx24_items:
            # unit_price가 7.xx대여야 함 (120.xx나 84.xx면 잘못된 것)
            unit_price = sodwx24_items[0].unit_price
            assert unit_price is not None, "unit_price가 None임"
            assert 5.0 <= unit_price <= 15.0, \
                f"unit_price가 Your Price가 아님: {unit_price} (기대: ~7.00)"

    def test_sftwb14_ocr_correction(self, sample_001_text):
        """SFTWBIY가 SFTWB14로 보정되어야 함"""
        result = parse_sng_invoice(sample_001_text)

        item_codes = [item.item_code for item in result.line_items]

        # SFTWBIY가 아닌 SFTWB14가 있어야 함
        assert "SFTWBIY" not in item_codes, "SFTWBIY가 보정되지 않음"
        assert "SFTWB14" in item_codes, "SFTWB14가 감지되지 않음"


class TestNormalization:
    """라인 정규화 테스트"""

    def test_normalize_0g_to_og(self):
        """0G가 OG로 정규화되어야 함"""
        # TODO: normalize_line() 함수 구현 후 테스트
        pass

    def test_normalize_special_chars(self):
        """«, (, ) 같은 특수문자가 제거되어야 함"""
        # TODO: normalize_line() 함수 구현 후 테스트
        pass

    def test_normalize_comma_to_dot(self):
        """가격의 쉼표가 점으로 정규화되어야 함 (74,08 -> 74.08)"""
        # TODO: normalize_line() 함수 구현 후 테스트
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
