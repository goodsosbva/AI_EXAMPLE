"""웹 페이지 설명을 보고 필요한 테스트 함수를 고르는 Function Calling 예제."""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable


MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

TEST_CHECKS = {
    "test_page_load": ["정상 응답과 리다이렉트", "제목과 주요 콘텐츠 표시", "치명적인 로딩 오류"],
    "test_navigation": ["메뉴와 링크 이동", "현재 위치 표시", "뒤로 가기 후 상태 유지"],
    "test_form": ["필수값과 형식 검증", "정상 제출", "오류 메시지와 입력값 유지"],
    "test_authentication": ["정상·실패 로그인", "비로그인 접근 차단", "로그아웃과 세션 만료"],
    "test_accessibility": ["키보드 탐색과 포커스", "레이블·대체 텍스트", "제목 구조와 대비"],
    "test_responsive": ["360·768·1280px 배치", "가로 스크롤 여부", "터치 영역과 메뉴 동작"],
}


def _result(name: str, page: str, reason: str) -> dict:
    page, reason = page.strip(), reason.strip()
    if not page or len(page) > 2048:
        raise ValueError("page는 1~2048자로 입력해야 합니다.")
    if not reason or len(reason) > 500:
        raise ValueError("reason은 1~500자로 입력해야 합니다.")
    return {"test_function": name, "page": page, "reason": reason, "checks": TEST_CHECKS[name]}


def test_page_load(page: str, reason: str) -> dict:
    return _result("test_page_load", page, reason)


def test_navigation(page: str, reason: str) -> dict:
    return _result("test_navigation", page, reason)


def test_form(page: str, reason: str) -> dict:
    return _result("test_form", page, reason)


def test_authentication(page: str, reason: str) -> dict:
    return _result("test_authentication", page, reason)


def test_accessibility(page: str, reason: str) -> dict:
    return _result("test_accessibility", page, reason)


def test_responsive(page: str, reason: str) -> dict:
    return _result("test_responsive", page, reason)


FUNCTIONS: dict[str, Callable[..., dict]] = {
    function.__name__: function
    for function in (
        test_page_load,
        test_navigation,
        test_form,
        test_authentication,
        test_accessibility,
        test_responsive,
    )
}


def _tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "string", "description": "테스트할 페이지 URL, 경로 또는 이름"},
                "reason": {"type": "string", "description": "이 테스트 함수가 필요한 구체적인 이유"},
            },
            "required": ["page", "reason"],
            "additionalProperties": False,
        },
    }


TOOLS = [
    _tool("test_page_load", "모든 페이지의 로딩과 핵심 콘텐츠 표시를 확인합니다."),
    _tool("test_navigation", "메뉴, 링크, 탭 등 페이지 이동 요소가 있을 때 사용합니다."),
    _tool("test_form", "검색, 문의, 회원가입 등 일반 입력 폼이 있을 때 사용합니다. 로그인은 제외합니다."),
    _tool("test_authentication", "로그인, 로그아웃, 권한 또는 세션 기능이 있을 때 사용합니다."),
    _tool("test_accessibility", "키보드, 레이블, 대체 텍스트 등 기본 접근성을 확인합니다."),
    _tool("test_responsive", "모바일·태블릿·데스크톱 레이아웃을 확인해야 할 때 사용합니다."),
]


def choose_web_tests(page: str, description: str) -> tuple[list[dict], str]:
    """모델이 테스트 함수를 고르게 하고 선택 결과를 반환합니다."""
    page, description = page.strip(), description.strip()
    if not page or len(page) > 2048:
        raise ValueError("page는 1~2048자로 입력해야 합니다.")
    if not description or len(description) > 5000:
        raise ValueError("description은 1~5000자로 입력해야 합니다.")

    from openai import OpenAI

    client = OpenAI()
    inputs: list = [
        {
            "role": "user",
            "content": f"페이지: {page}\n설명: {description}",
        }
    ]
    response = client.responses.create(
        model=MODEL,
        instructions=(
            "당신은 웹 QA 담당자입니다. 설명에 근거해 꼭 필요한 테스트 함수만 선택하세요. "
            "test_page_load와 test_accessibility는 항상 선택하고, 나머지는 해당 기능이 명시된 경우에만 선택하세요. "
            "없는 함수는 만들지 마세요."
        ),
        input=inputs,
        tools=TOOLS,
        tool_choice="required",
        parallel_tool_calls=True,
    )

    inputs += response.output
    selected = []
    for item in response.output:
        if item.type != "function_call":
            continue
        function = FUNCTIONS.get(item.name)
        if function is None:
            raise ValueError(f"허용되지 않은 함수입니다: {item.name}")
        try:
            result = function(**json.loads(item.arguments))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            result = {"test_function": item.name, "error": str(error)}
        selected.append(result)
        inputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            }
        )

    if not selected:
        raise RuntimeError("모델이 테스트 함수를 선택하지 않았습니다.")

    final = client.responses.create(
        model=MODEL,
        instructions="선택된 테스트 함수와 이유를 한국어로 짧게 요약하세요.",
        input=inputs,
    )
    return selected, final.output_text


def self_test() -> None:
    assert set(FUNCTIONS) == set(TEST_CHECKS)
    result = test_form("/contact", "문의 폼이 있음")
    assert result["test_function"] == "test_form" and len(result["checks"]) == 3
    try:
        test_form("", "이유")
    except ValueError:
        pass
    else:
        raise AssertionError("빈 page가 거부되지 않았습니다.")
    print("self-test 통과")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", nargs="?", help="페이지 URL, 경로 또는 이름")
    parser.add_argument("description", nargs="?", help="페이지 기능 설명")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.page or not args.description:
        parser.error("page와 description을 입력하세요.")

    selected, summary = choose_web_tests(args.page, args.description)
    print("\n[선택된 함수]")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print("\n[요약]")
    print(summary)


if __name__ == "__main__":
    main()
