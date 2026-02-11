"""
PII(개인정보) 검출 테스트 케이스 통합 모듈 (총 100개)

사용법:
    from pii_test_cases import ALL_TEST_CASES

    for tc in ALL_TEST_CASES:
        print(f"[{tc['id']}] {tc['category']} ({tc['difficulty']}) - {tc['intent'][:50]}...")

카테고리 구성:
    Part 1 (TC001~TC022): 이름, 주소
    Part 2 (TC023~TC056): 주민등록번호, 여권번호/운전면허번호, 이메일, IP주소
    Part 3 (TC057~TC072): 전화번호, 금융정보(계좌번호/카드번호)
    Part 4 (TC073~TC087): 복합 PII (여러 유형 혼합)
    Part 5 (TC088~TC100): False Positive(오탐지 검증), 난독화/변형

난이도 분포:
    EASY   : 표준 형식, 명확한 패턴
    MEDIUM : 문맥 속 포함, 형식 변형
    HARD   : 난독화, 비정형, 오탐 유사 패턴
"""

from .pii_test_cases import TEST_CASES as _part1
from .pii_test_cases_part2 import TEST_CASES_PART2 as _part2
from .pii_test_cases_part3 import TEST_CASES_PART3 as _part3
from .pii_test_cases_part4 import TEST_CASES_PART4 as _part4
from .pii_test_cases_part5 import TEST_CASES_PART5 as _part5

ALL_TEST_CASES = _part1 + _part2 + _part3 + _part4 + _part5

# 카테고리별 접근용 딕셔너리
BY_CATEGORY = {}
for _tc in ALL_TEST_CASES:
    _cat = _tc["category"]
    BY_CATEGORY.setdefault(_cat, []).append(_tc)

# 난이도별 접근용 딕셔너리
BY_DIFFICULTY = {"EASY": [], "MEDIUM": [], "HARD": []}
for _tc in ALL_TEST_CASES:
    BY_DIFFICULTY[_tc["difficulty"]].append(_tc)

# ID로 개별 조회
BY_ID = {_tc["id"]: _tc for _tc in ALL_TEST_CASES}
