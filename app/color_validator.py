"""
색상 검증 모듈

Rails API에서 유효한 색상 목록을 가져와서 캐시합니다.
색상 영역 감지 시 이 목록을 사용하여 유효성 검증을 수행합니다.
"""
import os
import logging
import httpx
from typing import Set, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 환경변수
RAILS_API_URL = os.getenv("RAILS_API_URL", "https://hairsearcherapp.onrender.com")
COMPANY_CODE = os.getenv("COMPANY_CODE", "SNG")

# 캐시
_color_cache: Set[str] = set()
_cache_updated_at: Optional[datetime] = None
_cache_ttl = timedelta(hours=1)  # 1시간마다 갱신


def get_valid_colors() -> Set[str]:
    """
    유효한 색상 목록 반환 (캐시 사용)

    Returns:
        색상 코드 Set (대문자)
    """
    global _color_cache, _cache_updated_at

    # 캐시가 유효하면 반환
    if _cache_updated_at and datetime.now() - _cache_updated_at < _cache_ttl:
        return _color_cache

    # API에서 새로 가져오기
    try:
        colors = fetch_colors_from_api()
        if colors:
            _color_cache = colors
            _cache_updated_at = datetime.now()
            logger.info(f"색상 목록 갱신: {len(colors)}개")
            return _color_cache
    except Exception as e:
        logger.warning(f"색상 목록 가져오기 실패: {e}")

    # 캐시가 있으면 오래되어도 사용
    if _color_cache:
        return _color_cache

    # 캐시도 없으면 빈 셋 반환
    return set()


def fetch_colors_from_api() -> Set[str]:
    """
    Rails API에서 색상 목록 가져오기

    Returns:
        색상 코드 Set (대문자)
    """
    url = f"{RAILS_API_URL}/api/v1/colors?company_code={COMPANY_CODE}"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            colors = data.get("colors", [])
            # 대문자로 정규화
            return {c.upper().strip() for c in colors if c}

    except httpx.RequestError as e:
        logger.error(f"Rails API 요청 실패: {e}")
        raise
    except Exception as e:
        logger.error(f"색상 목록 파싱 실패: {e}")
        raise


def is_valid_color(color: str) -> bool:
    """
    색상 코드가 유효한지 확인

    Args:
        color: 색상 코드 (원본)

    Returns:
        유효하면 True
    """
    if not color:
        return False

    valid_colors = get_valid_colors()

    # 캐시가 비어있으면 기본 패턴 매칭 사용
    if not valid_colors:
        return _is_valid_color_pattern(color)

    color_upper = color.upper().strip()

    # 정확히 일치
    if color_upper in valid_colors:
        return True

    # 변형 확인 (IB→1B, OB→0B 등)
    variants = generate_color_variants(color_upper)
    for variant in variants:
        if variant in valid_colors:
            return True

    return False


def generate_color_variants(color: str) -> list:
    """
    OCR 오인식을 고려한 색상 변형 생성

    Args:
        color: 색상 코드 (대문자)

    Returns:
        변형 목록
    """
    variants = []

    # I ↔ 1 변환
    if 'I' in color:
        variants.append(color.replace('I', '1'))
    if '1' in color:
        variants.append(color.replace('1', 'I'))

    # O ↔ 0 변환
    if 'O' in color:
        variants.append(color.replace('O', '0'))
    if '0' in color:
        variants.append(color.replace('0', 'O'))

    # IB → 1B (첫 글자)
    if len(color) >= 2:
        if color[0] == 'I':
            variants.append('1' + color[1:])
        elif color[0] == 'O':
            variants.append('0' + color[1:])
        elif color[0] == 'L':
            variants.append('1' + color[1:])
        elif color[0] == 'S':
            variants.append('5' + color[1:])

    return variants


def _is_valid_color_pattern(color: str) -> bool:
    """
    기본 색상 패턴 매칭 (캐시 없을 때 fallback)

    유효 패턴:
    - 숫자만: 1, 2, 27, 30, 613
    - 숫자+알파벳: 1B, 99J
    - 알파벳+숫자: T27, OT30, P1B/30
    - 알파벳만: NATURAL, GREY, COPPER
    """
    import re

    if not color:
        return False

    color = color.upper().strip()

    # 길이 제한
    if len(color) < 1 or len(color) > 15:
        return False

    # 유효한 문자만 (알파벳, 숫자, 슬래시, 하이픈)
    if not re.match(r'^[A-Z0-9/\-]+$', color):
        return False

    return True


def preload_colors():
    """
    서비스 시작 시 색상 목록 미리 로드
    """
    try:
        colors = get_valid_colors()
        logger.info(f"색상 목록 preload 완료: {len(colors)}개")
    except Exception as e:
        logger.warning(f"색상 목록 preload 실패: {e}")
