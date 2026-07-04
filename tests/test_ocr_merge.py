"""
OCR Merge 테스트

bbox 기반 OCR 및 block merge 로직 검증:
1. ocr_with_bbox: image_to_data로 bbox 좌표 획득
2. detect_color_regions_bbox: 색상 영역 감지
3. merge_by_replacement: 원본 라인을 재OCR 결과로 교체

핵심 검증:
- merge가 색상 라인만 교체하는지 (item/price/footer 보존)
- 여러 아이템에 걸친 color region이 과치환되지 않는지
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# pytesseract 없이 테스트 가능하도록 ocr_models에서 직접 import
from app.ocr_models import (
    OcrLine,
    ColorRegionResult,
    OcrConfig,
    DEFAULT_OCR_CONFIG,
    COLOR_REGION_OCR_CONFIG,
    merge_by_replacement,
    detect_color_regions_bbox,
)


class TestOcrConfig:
    """OcrConfig 설정 테스트"""

    def test_default_config_values(self):
        """기본 설정 값 확인"""
        config = DEFAULT_OCR_CONFIG

        assert config.scale_factor == 2.0
        assert config.contrast == 1.8
        assert config.threshold == 150
        assert config.sharpen_passes == 1
        assert config.psm == 6
        assert config.char_whitelist == ""

    def test_color_region_config_values(self):
        """색상 영역 전용 설정 값 확인"""
        config = COLOR_REGION_OCR_CONFIG

        assert config.scale_factor == 3.0  # 더 높은 확대
        assert config.contrast == 2.5      # 더 강한 대비
        assert config.threshold == 140     # 더 낮은 임계값
        assert config.sharpen_passes == 2  # 2회 샤프닝
        assert config.psm == 6
        assert "0123456789" in config.char_whitelist  # 숫자 포함
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" in config.char_whitelist  # 대문자 포함


class TestMergeByReplacement:
    """merge_by_replacement 함수 테스트"""

    def test_no_color_results_returns_original(self):
        """색상 결과 없으면 원본 텍스트 반환"""
        ocr_lines = [
            OcrLine(text="Invoice NO. 1234567890", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="STEST01 OG WATER CURL", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="7.00", y_top=120, y_bottom=140, confidence=90.0),
        ]

        result = merge_by_replacement(ocr_lines, [])

        assert "Invoice NO. 1234567890" in result
        assert "STEST01" in result
        assert "7.00" in result

    def test_replaces_color_lines_only(self):
        """색상 라인만 교체, 다른 라인 보존"""
        ocr_lines = [
            OcrLine(text="Invoice Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="STEST01 OG ITEM", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="7.00", y_top=120, y_bottom=140, confidence=90.0),
            OcrLine(text="IB - 3 2 - 5", y_top=140, y_bottom=160, confidence=50.0),  # 색상 라인 (저품질)
            OcrLine(text="Footer Info", y_top=200, y_bottom=220, confidence=90.0),
        ]

        # 색상 라인(인덱스 3)을 재OCR 결과로 교체
        color_results = [
            ColorRegionResult(
                region_text="1B - 3\n2 - 5",  # 고품질 재OCR 결과
                y_start=140,
                y_end=160,
                confidence=0.8,
                original_lines=[3]  # 인덱스 3 교체
            )
        ]

        result = merge_by_replacement(ocr_lines, color_results)

        # 헤더, 아이템, 가격, 푸터 보존
        assert "Invoice Header" in result
        assert "STEST01" in result
        assert "7.00" in result
        assert "Footer Info" in result

        # 원본 저품질 텍스트 제거, 고품질 결과로 교체
        assert "IB - 3 2 - 5" not in result  # 원본 제거
        assert "1B - 3" in result  # 교체 결과

    def test_replaces_multiple_color_lines(self):
        """여러 줄의 색상 라인 교체"""
        ocr_lines = [
            OcrLine(text="Item Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="Color Line 1", y_top=100, y_bottom=120, confidence=50.0),
            OcrLine(text="Color Line 2", y_top=120, y_bottom=140, confidence=50.0),
            OcrLine(text="Color Line 3", y_top=140, y_bottom=160, confidence=50.0),
            OcrLine(text="Footer", y_top=200, y_bottom=220, confidence=90.0),
        ]

        # 인덱스 1,2,3 모두 하나의 재OCR 블록으로 교체
        color_results = [
            ColorRegionResult(
                region_text="Replaced Color Block",
                y_start=100,
                y_end=160,
                confidence=0.8,
                original_lines=[1, 2, 3]
            )
        ]

        result = merge_by_replacement(ocr_lines, color_results)

        # 헤더, 푸터 보존
        assert "Item Header" in result
        assert "Footer" in result

        # 원본 색상 라인들 제거
        assert "Color Line 1" not in result
        assert "Color Line 2" not in result
        assert "Color Line 3" not in result

        # 교체 결과 포함
        assert "Replaced Color Block" in result

    def test_preserves_line_order(self):
        """라인 순서 보존"""
        ocr_lines = [
            OcrLine(text="Line 1", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="Color", y_top=50, y_bottom=70, confidence=50.0),
            OcrLine(text="Line 3", y_top=100, y_bottom=120, confidence=90.0),
        ]

        color_results = [
            ColorRegionResult(
                region_text="Replaced",
                y_start=50,
                y_end=70,
                confidence=0.8,
                original_lines=[1]
            )
        ]

        result = merge_by_replacement(ocr_lines, color_results)
        lines = result.split('\n')

        # 순서 확인
        line1_idx = next(i for i, l in enumerate(lines) if "Line 1" in l)
        replaced_idx = next(i for i, l in enumerate(lines) if "Replaced" in l)
        line3_idx = next(i for i, l in enumerate(lines) if "Line 3" in l)

        assert line1_idx < replaced_idx < line3_idx


class TestDetectColorRegionsBbox:
    """detect_color_regions_bbox 함수 테스트"""

    def test_no_color_pattern_returns_empty(self):
        """색상 패턴 없으면 빈 리스트 반환"""
        ocr_lines = [
            OcrLine(text="Invoice Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="STEST01 OG ITEM", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="7.00", y_top=120, y_bottom=140, confidence=90.0),
        ]

        result = detect_color_regions_bbox(ocr_lines, (1000, 500))

        assert result == []

    def test_detects_color_pattern(self):
        """색상-수량 패턴 감지"""
        ocr_lines = [
            OcrLine(text="Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="1B - 3 2 - 5", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="Footer", y_top=200, y_bottom=220, confidence=90.0),
        ]

        result = detect_color_regions_bbox(ocr_lines, (1000, 500))

        assert len(result) >= 1
        # 결과는 (y_start, y_end, [line_indices]) 튜플
        y_start, y_end, line_indices = result[0]
        assert 1 in line_indices  # 인덱스 1이 색상 라인

    def test_groups_adjacent_color_lines(self):
        """인접한 색상 라인 그룹화"""
        ocr_lines = [
            OcrLine(text="Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="1B - 3", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="2 - 5", y_top=125, y_bottom=145, confidence=90.0),  # 인접 (5px 차이)
            OcrLine(text="30 - 2", y_top=150, y_bottom=170, confidence=90.0),  # 인접
            OcrLine(text="Footer", y_top=300, y_bottom=320, confidence=90.0),
        ]

        result = detect_color_regions_bbox(ocr_lines, (1000, 500))

        # 인접한 색상 라인들이 하나의 그룹으로 묶여야 함
        assert len(result) == 1
        _, _, line_indices = result[0]
        assert set(line_indices) == {1, 2, 3}

    def test_separates_distant_color_regions(self):
        """떨어진 색상 영역 분리"""
        ocr_lines = [
            OcrLine(text="Item 1", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="1B - 3", y_top=50, y_bottom=70, confidence=90.0),
            OcrLine(text="Item 2", y_top=200, y_bottom=220, confidence=90.0),  # 중간에 다른 라인
            OcrLine(text="2 - 5", y_top=250, y_bottom=270, confidence=90.0),  # 거리 먼 색상
        ]

        result = detect_color_regions_bbox(ocr_lines, (1000, 500))

        # 두 개의 분리된 색상 영역
        assert len(result) == 2

    def test_marks_lines_as_color_region(self):
        """색상 라인에 is_color_region 플래그 설정"""
        ocr_lines = [
            OcrLine(text="Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="1B - 3", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="Footer", y_top=200, y_bottom=220, confidence=90.0),
        ]

        detect_color_regions_bbox(ocr_lines, (1000, 500))

        # 색상 라인에 플래그 설정
        assert ocr_lines[0].is_color_region is False
        assert ocr_lines[1].is_color_region is True
        assert ocr_lines[2].is_color_region is False


class TestColorRegionSafety:
    """색상 영역 안전성 테스트 - item/price/footer 보존 확인"""

    def test_does_not_replace_item_lines(self):
        """아이템 라인이 색상 라인으로 오인되지 않아야 함"""
        # 아이템 코드에 숫자가 있어도 색상 패턴과 다름
        ocr_lines = [
            OcrLine(text="STEST14 OG WATER CURL 14\"", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="1B - 3", y_top=150, y_bottom=170, confidence=90.0),
        ]

        result = detect_color_regions_bbox(ocr_lines, (1000, 500))

        # 아이템 라인(0)은 색상 영역에 포함되지 않아야 함
        for _, _, line_indices in result:
            assert 0 not in line_indices

    def test_does_not_replace_price_lines(self):
        """가격 라인이 색상 라인으로 오인되지 않아야 함"""
        ocr_lines = [
            OcrLine(text="7.00 10.00 84.00", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="1B - 3", y_top=150, y_bottom=170, confidence=90.0),
        ]

        result = detect_color_regions_bbox(ocr_lines, (1000, 500))

        # 가격 라인(0)은 색상 영역에 포함되지 않아야 함
        for _, _, line_indices in result:
            assert 0 not in line_indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
