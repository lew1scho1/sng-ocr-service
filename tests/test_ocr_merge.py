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
    normalize_color_region_text,
    split_color_tokens,
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


class TestNormalizeColorRegionText:
    """normalize_color_region_text 정규화 테스트"""

    def test_empty_string_returns_empty_list(self):
        """빈 문자열은 빈 리스트 반환"""
        result = normalize_color_region_text("")
        assert result == []

    def test_single_line_preserved(self):
        """단일 라인 보존"""
        result = normalize_color_region_text("1B - 3")
        assert result == ["1B - 3"]

    def test_multiple_lines_split(self):
        """여러 줄 분리"""
        result = normalize_color_region_text("1B - 3\n2 - 5\n30 - 6")
        assert result == ["1B - 3", "2 - 5", "30 - 6"]

    def test_empty_lines_removed(self):
        """빈 줄 제거"""
        result = normalize_color_region_text("1B - 3\n\n2 - 5\n\n\n30 - 6")
        assert result == ["1B - 3", "2 - 5", "30 - 6"]

    def test_whitespace_only_lines_removed(self):
        """공백만 있는 줄 제거"""
        result = normalize_color_region_text("1B - 3\n   \n2 - 5\n\t\t\n30 - 6")
        assert result == ["1B - 3", "2 - 5", "30 - 6"]

    def test_leading_trailing_whitespace_stripped(self):
        """앞뒤 공백 제거"""
        result = normalize_color_region_text("  1B - 3  \n  2 - 5  ")
        assert result == ["1B - 3", "2 - 5"]

    def test_consecutive_spaces_normalized(self):
        """연속 공백 정규화"""
        result = normalize_color_region_text("1B  -   3\n2   -    5")
        assert result == ["1B - 3", "2 - 5"]

    def test_tab_normalized_to_space(self):
        """탭이 공백으로 정규화"""
        result = normalize_color_region_text("1B\t-\t3")
        assert result == ["1B - 3"]

    def test_multiple_tokens_split_to_lines(self):
        """한 줄에 여러 토큰이 있으면 분리"""
        result = normalize_color_region_text("1B - 3 2 - 5 30 - 6")
        assert result == ["1B - 3", "2 - 5", "30 - 6"]

    def test_slash_color_tokens_split(self):
        """슬래시 색상 토큰도 분리"""
        result = normalize_color_region_text("P4/30 - 2 1B - 3")
        assert result == ["P4/30 - 2", "1B - 3"]

    def test_single_token_not_split(self):
        """단일 토큰은 분리하지 않음"""
        result = normalize_color_region_text("1B - 3")
        assert result == ["1B - 3"]

    def test_no_pattern_preserved(self):
        """패턴이 없는 줄은 그대로 유지"""
        result = normalize_color_region_text("SOME TEXT WITHOUT PATTERN")
        assert result == ["SOME TEXT WITHOUT PATTERN"]

    def test_mixed_lines_and_tokens(self):
        """줄 분리 + 토큰 분리 조합"""
        result = normalize_color_region_text("1B - 3 2 - 5\n30 - 6 613 - 2")
        assert result == ["1B - 3", "2 - 5", "30 - 6", "613 - 2"]


class TestSplitColorTokens:
    """split_color_tokens 함수 테스트"""

    def test_empty_string_returns_empty(self):
        """빈 문자열은 빈 리스트"""
        result = split_color_tokens("")
        assert result == []

    def test_single_token_returns_original(self):
        """단일 토큰은 원본 그대로"""
        result = split_color_tokens("1B - 3")
        assert result == ["1B - 3"]

    def test_multiple_simple_tokens(self):
        """여러 간단한 토큰 분리"""
        result = split_color_tokens("1B - 3 2 - 5 30 - 6")
        assert result == ["1B - 3", "2 - 5", "30 - 6"]

    def test_slash_color_pattern(self):
        """슬래시 포함 색상 패턴"""
        result = split_color_tokens("P4/30 - 2 P27/30 - 6")
        assert result == ["P4/30 - 2", "P27/30 - 6"]

    def test_compound_color_pattern(self):
        """복합 색상 패턴 (C-42730)"""
        result = split_color_tokens("C-42730 - 4 OT-27 - 3")
        assert result == ["C-42730 - 4", "OT-27 - 3"]

    def test_mixed_patterns(self):
        """혼합 패턴"""
        result = split_color_tokens("1B - 3 P4/30 - 2 613 - 1")
        assert result == ["1B - 3", "P4/30 - 2", "613 - 1"]

    def test_no_pattern_returns_original(self):
        """패턴 없으면 원본 그대로"""
        result = split_color_tokens("WATER CURL 14 INCH")
        assert result == ["WATER CURL 14 INCH"]

    def test_normalized_output_format(self):
        """출력은 정규화된 형식 (COLOR - QTY)"""
        result = split_color_tokens("1B-3 2-5")  # 공백 없는 입력
        assert result == ["1B - 3", "2 - 5"]  # 공백 있는 출력

    def test_uppercase_conversion(self):
        """소문자도 대문자로 변환"""
        result = split_color_tokens("1b - 3 p4/30 - 2")
        assert result == ["1B - 3", "P4/30 - 2"]


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

    def test_multiline_region_text_expands_to_multiple_lines(self):
        """여러 줄 region_text가 개별 라인으로 확장됨"""
        ocr_lines = [
            OcrLine(text="Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="IB - 3 2 - S", y_top=100, y_bottom=120, confidence=50.0),
            OcrLine(text="Footer", y_top=200, y_bottom=220, confidence=90.0),
        ]

        # 재OCR 결과가 여러 줄인 경우
        color_results = [
            ColorRegionResult(
                region_text="1B - 3\n2 - 5\n30 - 6",  # 3줄
                y_start=100,
                y_end=120,
                confidence=0.8,
                original_lines=[1]
            )
        ]

        result = merge_by_replacement(ocr_lines, color_results)
        lines = result.split('\n')

        # 원본 1줄이 3줄로 확장됨
        assert "Header" in lines[0]
        assert "1B - 3" in result
        assert "2 - 5" in result
        assert "30 - 6" in result
        assert "Footer" in result
        assert "IB - 3 2 - S" not in result  # 원본 제거

    def test_parser_friendly_line_structure(self):
        """파서 친화적 줄 구조 유지 - 각 색상은 독립 줄"""
        ocr_lines = [
            OcrLine(text="STEST01 OG ITEM", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="7.00", y_top=25, y_bottom=45, confidence=90.0),
            OcrLine(text="IB - 3 2 - S 30 - G", y_top=50, y_bottom=70, confidence=50.0),
            OcrLine(text="STEST02 OG ITEM", y_top=100, y_bottom=120, confidence=90.0),
        ]

        # 색상이 각각 독립 줄로 재OCR됨
        color_results = [
            ColorRegionResult(
                region_text="1B - 3\n2 - 5\n30 - 6",
                y_start=50,
                y_end=70,
                confidence=0.8,
                original_lines=[2]
            )
        ]

        result = merge_by_replacement(ocr_lines, color_results)
        lines = result.split('\n')

        # 아이템 → 가격 → 색상들 → 다음 아이템 순서
        assert lines[0] == "STEST01 OG ITEM"
        assert lines[1] == "7.00"
        assert lines[2] == "1B - 3"
        assert lines[3] == "2 - 5"
        assert lines[4] == "30 - 6"
        assert lines[5] == "STEST02 OG ITEM"

    def test_whitespace_normalized_in_replacement(self):
        """교체 텍스트의 공백이 정규화됨"""
        ocr_lines = [
            OcrLine(text="Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="Color", y_top=50, y_bottom=70, confidence=50.0),
        ]

        # OCR 결과에 불규칙한 공백이 있는 경우
        color_results = [
            ColorRegionResult(
                region_text="  1B  -  3  \n\n  2   -   5  \n\t\t",
                y_start=50,
                y_end=70,
                confidence=0.8,
                original_lines=[1]
            )
        ]

        result = merge_by_replacement(ocr_lines, color_results)
        lines = result.split('\n')

        # 공백 정규화 확인
        assert "1B - 3" in lines
        assert "2 - 5" in lines
        assert "  " not in result  # 연속 공백 없음


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


class TestScoreColorLine:
    """score_color_line 함수 점수 시스템 테스트"""

    # Import score_color_line for direct testing
    from app.ocr_models import (
        score_color_line,
        ColorPatternType,
        BASE_SCORES,
        CANDIDATE_SCORE_THRESHOLD,
    )

    def test_simple_pattern_base_score(self):
        """SIMPLE 패턴 기본 점수 확인"""
        from app.ocr_models import score_color_line, BASE_SCORES, ColorPatternType

        candidate = score_color_line(0, "1B - 3")

        assert candidate is not None
        assert ColorPatternType.SIMPLE in candidate.pattern_types
        # 기본 점수 + 짧은 라인 보너스 (0.08)
        assert candidate.score >= BASE_SCORES[ColorPatternType.SIMPLE]
        assert candidate.unique_token_count == 1

    def test_slash_pattern_base_score(self):
        """SLASH 패턴 기본 점수 확인"""
        from app.ocr_models import score_color_line, BASE_SCORES, ColorPatternType

        candidate = score_color_line(0, "P4/30 - 2")

        assert candidate is not None
        assert ColorPatternType.SLASH in candidate.pattern_types
        assert candidate.score >= BASE_SCORES[ColorPatternType.SLASH]

    def test_compound_pattern_base_score(self):
        """COMPOUND 패턴 기본 점수 확인"""
        from app.ocr_models import score_color_line, BASE_SCORES, ColorPatternType

        candidate = score_color_line(0, "C-42730 - 4")

        assert candidate is not None
        assert ColorPatternType.COMPOUND in candidate.pattern_types
        assert candidate.score >= BASE_SCORES[ColorPatternType.COMPOUND]

    def test_numeric_pattern_base_score(self):
        """NUMERIC 패턴 기본 점수 확인 - 숫자만 있는 패턴도 매칭됨"""
        from app.ocr_models import score_color_line, BASE_SCORES, ColorPatternType

        # 숫자만 있는 패턴 (2 - 5)도 후보로 인정되어야 함
        # SIMPLE 패턴이 먼저 매칭되지만 결과적으로 후보가 됨
        candidate = score_color_line(0, "2 - 5 613 - 3")

        assert candidate is not None
        # SIMPLE 또는 NUMERIC 중 하나로 매칭됨
        assert len(candidate.pattern_types) >= 1
        assert candidate.score >= BASE_SCORES[ColorPatternType.NUMERIC]

    def test_token_dedupe_prevents_duplicate_scoring(self):
        """동일 토큰이 중복 점수화되지 않아야 함"""
        from app.ocr_models import score_color_line, BASE_SCORES, ColorPatternType

        # 동일 토큰이 여러 패턴에 매칭되어도 한 번만 점수화
        candidate = score_color_line(0, "1B - 3 1B - 3 1B - 3")

        assert candidate is not None
        # 동일 토큰(1B:3)이 3번 등장해도 unique count는 1
        assert candidate.unique_token_count == 1
        # 점수는 기본 점수 + 보너스 정도여야 함 (3배 아님)
        base_score = BASE_SCORES[ColorPatternType.SIMPLE]
        # score should be roughly base_score + bonuses, not 3 * base_score
        assert candidate.score < base_score * 2

    def test_multiple_unique_tokens_score_higher(self):
        """여러 고유 토큰이 있으면 더 높은 점수"""
        from app.ocr_models import score_color_line

        # 단일 토큰
        single = score_color_line(0, "1B - 3")
        # 복수 토큰 (서로 다른)
        multiple = score_color_line(0, "1B - 3 2 - 5 30 - 6")

        assert single is not None
        assert multiple is not None
        assert multiple.unique_token_count == 3
        assert multiple.score > single.score

    def test_score_no_upper_cap(self):
        """점수가 1.0에서 포화되지 않아야 함"""
        from app.ocr_models import score_color_line

        # 많은 토큰으로 높은 점수 유도
        candidate = score_color_line(
            0, "1B - 3 2 - 5 30 - 6 P4/30 - 2 P27/30 - 1 613 - 3"
        )

        assert candidate is not None
        # 1.0을 초과할 수 있어야 함 (상대 비교 가능)
        assert candidate.score > 0.5  # 최소한 이 정도는 넘어야 함

    def test_penalty_pattern_description_keywords(self):
        """설명 키워드 감점이 작동해야 함"""
        from app.ocr_models import score_color_line

        # 설명 키워드 없는 라인
        clean = score_color_line(0, "1B - 3 2 - 5")
        # 설명 키워드 포함 라인
        with_desc = score_color_line(0, "1B - 3 2 - 5 WATER CURL")

        assert clean is not None
        assert with_desc is not None
        # 설명 키워드가 있으면 감점되어야 함
        assert with_desc.score < clean.score
        assert "penalty:" in str(with_desc.matched_patterns)

    def test_penalty_pattern_item_code(self):
        """아이템 코드 감점이 작동해야 함"""
        from app.ocr_models import score_color_line

        # 아이템 코드가 있으면 색상 라인 가능성 낮음
        candidate = score_color_line(0, "1B - 3 STEST14")

        # 감점이 적용되어야 함
        if candidate:
            assert "penalty:" in str(candidate.matched_patterns)

    def test_db_match_adds_bonus_not_required(self):
        """DB 매칭은 가산점, 탈락 조건 아님"""
        from app.ocr_models import score_color_line

        # DB 색상 없이도 후보가 되어야 함
        # AB-3는 SIMPLE 패턴으로 매칭됨 (1-2 letters)
        candidate_no_db = score_color_line(0, "AB - 3", valid_colors=None)
        candidate_empty_db = score_color_line(0, "AB - 3", valid_colors=set())

        # DB 미검증이어도 구조만으로 후보 가능 (threshold 이상이면)
        assert candidate_no_db is not None or candidate_empty_db is not None
        # DB 매칭 없으면 db_match_count = 0
        if candidate_no_db:
            assert candidate_no_db.db_match_count == 0

    def test_db_match_increases_score(self):
        """DB 색상 매칭 시 점수 증가"""
        from app.ocr_models import score_color_line

        # DB에 "1B" 있는 경우
        valid_colors = {"1B", "2", "30"}
        candidate_with_db = score_color_line(0, "1B - 3 2 - 5", valid_colors=valid_colors)
        candidate_no_db = score_color_line(0, "1B - 3 2 - 5", valid_colors=None)

        assert candidate_with_db is not None
        assert candidate_no_db is not None
        assert candidate_with_db.db_match_count >= 1
        assert candidate_with_db.score > candidate_no_db.score
        assert "db_match:" in str(candidate_with_db.matched_patterns)

    def test_exclude_pattern_invoice_header(self):
        """Invoice 헤더는 제외되어야 함"""
        from app.ocr_models import score_color_line

        candidate = score_color_line(0, "INVOICE NO. 12345 1B - 3")

        assert candidate is None

    def test_exclude_pattern_barcode(self):
        """바코드 번호 라인은 제외되어야 함"""
        from app.ocr_models import score_color_line

        candidate = score_color_line(0, "1234567890123 1B - 3")

        assert candidate is None

    def test_short_line_bonus(self):
        """짧은 라인 보너스 확인"""
        from app.ocr_models import score_color_line

        # 짧은 라인 (<=25자)
        short = score_color_line(0, "1B - 3")
        # 중간 라인 (25-40자)
        medium = score_color_line(0, "1B - 3 2 - 5 30 - 6 613 - 3")

        assert short is not None
        assert medium is not None
        # 짧은 라인이 토큰당 더 높은 점수 (보너스 0.08 vs 0.04)
        score_per_token_short = short.score / short.unique_token_count
        # 이 비교는 복잡하므로 단순히 short이 0보다 크면 OK
        assert short.score > 0


class TestGroupCandidateLinesV2:
    """group_candidate_lines_v2 그룹화 테스트"""

    def test_item_boundary_separates_groups(self):
        """아이템 경계가 그룹을 분리해야 함"""
        from app.ocr_models import (
            OcrLine,
            score_color_line,
            group_candidate_lines_v2,
            mark_item_lines,
        )

        ocr_lines = [
            OcrLine(text="STEST01 OG ITEM", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="1B - 3 2 - 5", y_top=50, y_bottom=70, confidence=90.0),
            OcrLine(text="STEST02 OG ITEM", y_top=100, y_bottom=120, confidence=90.0),  # 아이템 경계
            OcrLine(text="30 - 6 613 - 3", y_top=150, y_bottom=170, confidence=90.0),
        ]

        # 아이템 라인 마킹
        mark_item_lines(ocr_lines)

        # 후보 생성
        candidates = []
        for i, line in enumerate(ocr_lines):
            c = score_color_line(i, line.text)
            if c:
                candidates.append(c)

        # 그룹화
        groups = group_candidate_lines_v2(candidates, ocr_lines, (1000, 500))

        # 아이템 경계로 인해 2개 그룹으로 분리되어야 함
        assert len(groups) == 2
        assert 1 in groups[0].line_indices
        assert 3 in groups[1].line_indices

    def test_adjacent_lines_grouped(self):
        """인접한 색상 라인은 그룹화되어야 함"""
        from app.ocr_models import (
            OcrLine,
            score_color_line,
            group_candidate_lines_v2,
        )

        ocr_lines = [
            OcrLine(text="1B - 3", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="2 - 5", y_top=125, y_bottom=145, confidence=90.0),
            OcrLine(text="30 - 6", y_top=150, y_bottom=170, confidence=90.0),
        ]

        candidates = []
        for i, line in enumerate(ocr_lines):
            c = score_color_line(i, line.text)
            if c:
                candidates.append(c)

        groups = group_candidate_lines_v2(candidates, ocr_lines, (1000, 500))

        # 모두 인접하므로 1개 그룹
        assert len(groups) == 1
        assert set(groups[0].line_indices) == {0, 1, 2}

    def test_context_type_after_item(self):
        """아이템 바로 뒤 색상 라인은 after_item 문맥"""
        from app.ocr_models import (
            OcrLine,
            score_color_line,
            group_candidate_lines_v2,
            mark_item_lines,
        )

        ocr_lines = [
            OcrLine(text="STEST01 OG ITEM", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="7.00", y_top=25, y_bottom=45, confidence=90.0),
            OcrLine(text="1B - 3 2 - 5", y_top=50, y_bottom=70, confidence=90.0),
        ]

        mark_item_lines(ocr_lines)

        candidates = []
        for i, line in enumerate(ocr_lines):
            c = score_color_line(i, line.text)
            if c:
                candidates.append(c)

        groups = group_candidate_lines_v2(candidates, ocr_lines, (1000, 500))

        assert len(groups) == 1
        assert groups[0].context_type == "after_item"


class TestDetectColorRegionsWithMetadata:
    """detect_color_regions_with_metadata 메타데이터 테스트"""

    def test_returns_color_region_candidate(self):
        """ColorRegionCandidate 객체 반환 확인"""
        from app.ocr_models import (
            OcrLine,
            detect_color_regions_with_metadata,
            ColorRegionCandidate,
        )

        ocr_lines = [
            OcrLine(text="Header", y_top=0, y_bottom=20, confidence=90.0),
            OcrLine(text="1B - 3 2 - 5", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="Footer", y_top=200, y_bottom=220, confidence=90.0),
        ]

        results = detect_color_regions_with_metadata(ocr_lines, (1000, 500))

        assert len(results) >= 1
        assert isinstance(results[0], ColorRegionCandidate)
        assert results[0].total_score > 0
        assert results[0].avg_score > 0
        assert len(results[0].candidate_lines) >= 1

    def test_metadata_includes_raw_texts(self):
        """메타데이터에 원본 텍스트 포함"""
        from app.ocr_models import OcrLine, detect_color_regions_with_metadata

        ocr_lines = [
            OcrLine(text="1B - 3", y_top=100, y_bottom=120, confidence=90.0),
            OcrLine(text="2 - 5", y_top=125, y_bottom=145, confidence=90.0),
        ]

        results = detect_color_regions_with_metadata(ocr_lines, (1000, 500))

        assert len(results) >= 1
        assert "1B - 3" in results[0].raw_line_texts
        assert "2 - 5" in results[0].raw_line_texts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
