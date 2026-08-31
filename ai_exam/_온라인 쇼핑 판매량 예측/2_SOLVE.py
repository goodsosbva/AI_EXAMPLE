---1.

# 데이터 분석의 시작점으로 필요한 라이브러리를 임포트하고,
# 제품 정보와 고객 통계 데이터를 데이터프레임으로 불러옴
# 형식이 서로 다른 파일(JSON/CSV)을 다루는 기본 역량을 확인

# 머신러닝과 데이터 처리에 필요한 필수 라이브러리 임포트
import sklearn as sk  # 사이킷런 전체를 sk 별칭으로 임포트
import pandas as pd   # 판다스를 pd 별칭으로 임포트

# 제품 정보 JSON 로딩
# JSON 파일을 데이터프레임으로 로딩
df_product = pd.read_json('product_info.json')

# 고객 인구통계 CSV 로딩
# CSV 파일을 데이터프레임으로 로딩
df_customer = pd.read_csv('customer_demographics.csv')

---2.

# 공통 키를 기준으로 데이터를 결합하고, df.info()와 df.head()로 데이터 상태를 빠르게 점검
# 이후 전처리 계획 수립을 위한 기초 탐색 단계

# ProductID 기준으로 inner join -> 두 데이터에 모두 존재하는 제품만 남김
df = pd.merge(df_product, df_customer, on='ProductID', how='inner')

# 데이터프레임 기본 정보 확인 (열 이름, 타입, 결측치 개요)
df.info()

# 상위 5행 미리보기로 데이터 전반 구조 파악
df.head()


---3.

# 범주형 분포를 시각화해 이상/불균형을 확인한 뒤 의미가 약한 범주('Misc')를 제거하여 학습 품질을 높임
# 삭제 전후 분포 비교로 정제 효과를 검증

# 시각화 라이브러리 임포트
import matplotlib.pyplot as plt
import seaborn as sns

# (정제 전) 카테고리 분포 시각화
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Category')
plt.title('Distribution of Category (Before Cleaning)')
plt.xticks(rotation=15)
plt.show()


# 'Misc' 카테고리 제거 및 삭제 수 계산
initial_rows = df.shape[0]                 # 삭제 전 전체 행 수
df = df[df['Category'] != 'Misc'].copy()   # 'Misc' 제거
답안03 = initial_rows - df.shape[0]        # 삭제된 행 수 저장

# (정제 후) 카테고리 분포 시각화
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Category')
plt.title('Distribution of Category (After Cleaning)')
plt.xticks(rotation=15)
plt.show()

# 확인 출력
print(f"삭제된 행의 개수 (답안03): {답안03}")

---4

# jointplot을 통해 이변량 분포를 확인하고, 비즈니스 규칙에 근거한 임계값(Price ≥ 5000)으로 이상치를 정의해 개수를 집계

# 가격과 판매량의 분포 및 상관 구조 탐색
sns.jointplot(data=df, x='Price', y='MonthlySales')
plt.suptitle('Relationship between Price and MonthlySales', y=1.02)
plt.show()

# 규칙 기반 이상치(Price >= 5000) 개수 산출
답안04 = df[df['Price'] >= 5000].shape[0]
print(f"Price가 5000 이상인 이상치 행의 개수 (답안04): {답안04}")

---5

# 이상치 행을 제거해 회귀 계수 왜곡을 줄이고, ID 컬럼 제거로 데이터 누수 및 과적합 가능성을 낮춤

# 이상치 제거 및 ID 컬럼 제거
df_temp = df[df['Price'] < 5000].copy()   # 이상치 제거
df_temp.drop('ProductID', axis=1, inplace=True)  # 식별자 열 제거

# 확인 출력
print("df_temp head after outlier and ProductID removal:")
print(df_temp.head())


---6

# 결측치가 있는 관측치를 제거해 학습 안정성을 높이고, 이진 범주 변수를 정수로 변환해 모델 입력으로 사용 가능하게함

# 결측치 분포 확인
print("결측치 확인 (df_temp):")
print(df_temp.isnull().sum())

# 결측치 행 삭제 전 행 수 기록
initial_rows_na = df_temp.shape[0]

# 결측치가 포함된 행 제거 -> df_na
df_na = df_temp.dropna().copy()

# 삭제된 행 수 계산
답안06 = initial_rows_na - df_na.shape[0]

# 이진 범주형 컬럼을 정수형으로 매핑(Yes=1, No=0)
df_na['PromotionApplied'] = df_na['PromotionApplied'].map({'Yes': 1, 'No': 0})

# 확인 출력
print("\n결측치 삭제 후 df_na info:")
df_na.info()
print(f"\n삭제된 결측치 행의 개수 (답안06): {답안06}")

# 다른 풀이

df_temp.isnull().sum()

prev = df_temp.shape[0]
print(prev)

df_na = df_temp.dropna()
print(df_na.shape[0])
답안06 = prev - df_na.shape[0]
print(답안06)

new_pro = []
for pro in df_na['PromotionApplied']:
  if pro == 'Yes':
    new_pro.append(1)
  else:
    new_pro.append(0)

df_na['PromotionApplied'] = new_pro
df_na.info()


--- 7

# 범주형을 더미 변수로 변환해 회귀/트리 모델이 처리할 수 있도록 함
# drop_first=True로 기준 범주를 제거해 더미 변수 함정을 완화

# 학습 단순화를 위해 Category와 타깃만 유지
df_del = df_na[['Category', 'MonthlySales']].copy()

# 범주형(Category) -> 원-핫 인코딩 (첫 범주 드롭으로 다중공선성 완화)
df_encoded = pd.get_dummies(df_del, columns=['Category'], drop_first=True)

# 확인 출력
print("원-핫 인코딩 후 컬럼:", df_encoded.columns.tolist())
print(df_encoded.head())

--- 8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# 피처/타깃 분리
X = df_encoded.drop('MonthlySales', axis=1)
y = df_encoded['MonthlySales']


# 데이터 분할 (재현성 확보)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 스케일링(트리 모델에 필수는 아니나, 다른 모델과의 파이프라인 일관성 유지)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터 기준으로 적합/변환
X_valid_scaled = scaler.transform(X_valid)      # 검증 데이터는 변환만 수행


# 형태 확인
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_valid_scaled shape:", X_valid_scaled.shape)


--- 9

# 배깅 기반 앙상블인 랜덤포레스트는 비선형 관계와 상호작용을 잘 포착하며 기본 성능이 안정적
from sklearn.ensemble import RandomForestRegressor

# 랜덤포레스트 회귀 모델 생성
rf_model = RandomForestRegressor(random_state=42)

# 모델 학습
rf_model.fit(X_train_scaled, y_train)

print("RandomForest 학습 완료")


--- 10

# 부스팅 기반 모델은 편향을 순차적으로 줄여나가며 복잡한 결정 경계를 학습
# 하이퍼파라미터 튜닝 여지가 큼
from sklearn.ensemble import GradientBoostingRegressor


# 그래디언트 부스팅 회귀 모델 생성
gb_model = GradientBoostingRegressor(random_state=42)

# 모델 학습
gb_model.fit(X_train_scaled, y_train)

print("GradientBoosting 학습 완료")


--- 11

# MAE는 예측 오차의 절대값 평균으로 해석이 직관적
# 값이 낮을수록 예측력이 높음을 의미

from sklearn.metrics import mean_absolute_error


# 검증 세트 예측
y_pred_rf = rf_model.predict(X_valid_scaled)
y_pred_gb = gb_model.predict(X_valid_scaled)


# MAE 산출
rf_mae = mean_absolute_error(y_valid, y_pred_rf)
gb_mae = mean_absolute_error(y_valid, y_pred_gb)


# 더 낮은 MAE를 보이는 모델 선택
답안11 = 'RandomForest' if rf_mae < gb_mae else 'GradientBoosting'

# 결과 출력
print(f"RandomForest MAE: {rf_mae:.4f}")
print(f"GradientBoosting MAE: {gb_mae:.4f}")
print(f"성능 좋은 모델 (답안11): {답안11}")

---12

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 재현성 고정
tf.random.set_seed(1)

# DNN 회귀 모델 설계: 입력 차원 = X_train_scaled의 열 개수
input_dim = X_train_scaled.shape[1]

dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),  # 1층 은닉
    Dropout(0.2),                                            # 과적합 방지
    Dense(32, activation='relu'),                            # 2층 은닉
    Dense(1)                                                 # 회귀 출력층
])

# 컴파일: 회귀 문제 -> 손실 mse, 옵티마이저 adam
dl_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# 모델 학습 (검증 세트로 성능 모니터링)
history_dl = dl_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    epochs=20,
    batch_size=16,
    verbose=0
)

print("DNN(MLP) 모델 학습 완료")

--- 13

import matplotlib.pyplot as plt

# 학습/검증 MSE 추출
train_mse = history_dl.history['mse']
val_mse = history_dl.history['val_mse']
epochs = range(1, len(train_mse) + 1)

# 학습 곡선 시각화
plt.figure(figsize=(8,5))
plt.plot(epochs, train_mse, label='Training MSE')
plt.plot(epochs, val_mse, label='Validation MSE')
plt.title('DNN Model Training vs Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

--- 14

from sklearn.metrics import mean_absolute_error

# DNN 모델 예측 및 MAE 산출
y_pred_dl = dl_model.predict(X_valid_scaled).flatten()
dl_mae = mean_absolute_error(y_valid, y_pred_dl)

# 기존 머신러닝 모델 MAE는 이전 단계에서 계산됨 (rf_mae, gb_mae)
print(f"RandomForest MAE: {rf_mae:.4f}")
print(f"GradientBoosting MAE: {gb_mae:.4f}")
print(f"DNN MAE: {dl_mae:.4f}")

# 세 모델 MAE 비교 후 최종 선택
min_mae = min(rf_mae, gb_mae, dl_mae)
if min_mae == rf_mae:
    답안14 = 'RandomForest'
elif min_mae == gb_mae:
    답안14 = 'GradientBoosting'
else:
    답안14 = 'DeepLearning (DNN)'

print(f"최종 선택 모델 (답안14): {답안14}")