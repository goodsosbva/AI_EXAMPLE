# AI Example

머신러닝과 LLM 실습을 개념별로 정리한 Jupyter/Colab 노트북 모음입니다.

## 분류

### Machine Learning

| 순서 | 개념 | 실습 |
| --- | --- | --- |
| 1 | 기초 | [첫 머신러닝](machine-learning/01-foundations/first-machine-learning.ipynb), [훈련/테스트 세트](machine-learning/01-foundations/train-test-split.ipynb), [데이터 전처리](machine-learning/01-foundations/data-preprocessing.ipynb) |
| 2 | 회귀 | [K-최근접 이웃 회귀](machine-learning/02-regression/knn-regression.ipynb), [선형 회귀](machine-learning/02-regression/linear-regression.ipynb) |
| 3 | 분류 | [로지스틱 회귀](machine-learning/03-classification/logistic-regression.ipynb), [확률적 경사 하강법](machine-learning/03-classification/stochastic-gradient-descent.ipynb) |
| 4 | 모델 개선 | [특성 공학과 규제](machine-learning/04-model-improvement/feature-engineering-and-regularization.ipynb), [교차 검증과 그리드 서치](machine-learning/04-model-improvement/cross-validation-and-grid-search.ipynb) |
| 5 | 트리 모델 | [결정 트리](machine-learning/05-tree-models/decision-tree.ipynb), [트리 앙상블](machine-learning/05-tree-models/tree-ensembles.ipynb) |

### LLM

| 순서 | 개념 | 실습 | Colab |
| --- | --- | --- | --- |
| 1 | OpenAI API·프롬프팅 | [OpenAI API를 이용한 서비스 개발](llm/01-openai-api/openai-api-service.ipynb) | [열기](https://colab.research.google.com/github/goodsosbva/AI_EXAMPLE/blob/main/llm/01-openai-api/openai-api-service.ipynb) |
| 2 | 토크나이저·챗 템플릿 | [챗 템플릿 이해하기](llm/02-chat-templates/chat-templates.ipynb) | [열기](https://colab.research.google.com/github/goodsosbva/AI_EXAMPLE/blob/main/llm/02-chat-templates/chat-templates.ipynb) |

## 사용 방법

1. GitHub에서 노트북을 열거나 LLM 표의 `Colab 열기`를 누릅니다.
2. 노트북의 셀을 위에서부터 순서대로 실행합니다.
3. OpenAI API 실습 전 Colab `Secrets`에 `OPENAI_API_KEY`를 등록합니다.

API 키나 비밀번호는 노트북에 직접 입력하지 않습니다.

## 새 실습 추가 규칙

- 가장 가까운 개념 폴더에 노트북 1개를 추가합니다.
- 파일명은 영문 소문자 `kebab-case.ipynb`를 사용합니다.
- 새 개념일 때만 폴더를 하나 추가합니다.
- 추가한 노트북은 이 README 표에 연결합니다.
