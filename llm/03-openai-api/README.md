# 웹 테스트 Function Calling 실행 방법

Git Bash가 `~/Desktop/side/AI_EXAMPLE/llm/03-openai-api` 위치이고 Python이 설치되어 있을 때 아래 두 줄을 그대로 실행합니다.

```bash
export OPENAI_API_KEY="새_API_키"
python web_test_function_calling.py "http://localhost:8000/create.html" "상품명, 카테고리, 가격, 재고를 입력해 새 상품을 등록하는 폼 페이지"
```

첫 줄의 `새_API_키`만 새로 발급한 실제 API 키로 바꾸면 됩니다.

`_` 앞에 `\`를 넣지 않습니다. URL도 `[주소](주소)`가 아니라 주소만 사용합니다.

```text
OPENAI_API_KEY
web_test_function_calling.py
http://localhost:8000/create.html
```

정상 실행 결과에는 `[선택된 함수]`와 `[요약]`이 출력됩니다.

## CRUD 실행 명령

```bash
# Read
python web_test_function_calling.py "http://localhost:8000/read.html" "상품 검색 폼과 상품 목록 표가 있는 조회 페이지"

# Update
python web_test_function_calling.py "http://localhost:8000/update.html" "기존 상품 정보를 수정하고 저장하는 폼 페이지"

# Delete
python web_test_function_calling.py "http://localhost:8000/delete.html" "상품 코드를 재입력하고 확인해야 영구 삭제할 수 있는 페이지"
```

## 자체 검사

```bash
python web_test_function_calling.py --self-test
```

## 골든 페이지 보기(선택)

별도 Git Bash 창에서 실행합니다.

```bash
python -m http.server 8000 --directory golden-pages
```

브라우저 주소는 `http://localhost:8000/create.html`입니다.

API 키는 코드나 저장소에 적지 말고, 공개된 기존 키는 폐기하십시오.
