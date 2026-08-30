---6

# df.isnull().sum()은 각 컬럼별 결측치의 개수를 알려줌
# df.dropna()는 결측치가 하나라도 있는 행을 제거하여 데이터를 정제
# df['Absences'].astype(int)는 Absences 컬럼의 데이터 타입을 정수형으로 변환
# 이는 결석 횟수와 같은 이산적인 데이터를 정확하게 표현하기 위해 필요

# 결측치 확인
print("결측치 확인 (변환 전):")
print(df_cleaned.isnull().sum())

# 결측치가 있는 모든 행 삭제
df_processed = df_cleaned.dropna().copy()

# Absences 컬럼을 정수형으로 변환
df_processed['Absences'] = df_processed['Absences'].astype(int)

print("\n결측치 처리 및 타입 변환 후:")
print(df_processed.isnull().sum())
print(df_processed.info())


---7

# 원-핫 인코딩은 머신러닝 모델이 범주형 데이터를 이해할 수 있도록 0과 1로 구성된 이진 벡터로 변환하는 기법
# pd.get_dummies()는 이를 자동으로 수행해주며, object 타입의 Gender와 ParentalEducation 컬럼을 새로운 컬럼들로 확장
# 예를 들어, Gender는 Gender_Female과 Gender_Male 컬럼으로 변환됨

# Age 컬럼 삭제
df_encoded = df_processed.drop('Age', axis=1).copy()

# object 타입 컬럼을 원-핫 인코딩
df_encoded = pd.get_dummies(df_encoded, columns=['Gender', 'ParentalEducation'])

print(df_encoded.head())
#print(df_encoded.info())


---8

# 데이터를 훈련용과 검증용으로 분리하는 것은 모델이 새로운 데이터에 대해 얼마나 잘 일반화되는지 평가하기 위해 필수적
# train_test_split은 이 작업을 효율적으로 수행
# RobustScaler는 이상치에 강건한 스케일링 방법으로, 데이터의 중앙값(median)과 IQR(사분위수 범위)을 사용하여 데이터를 스케일링
# fit_transform은 훈련 데이터에 대한 변환 규칙을 학습하고 적용하며, transform은 학습된 규칙을 검증 데이터에 그대로 적용합니다.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# Target(y)과 Feature(X) 분리
X = df_encoded.drop('FinalScore', axis=1)
y = df_encoded['FinalScore']


# 훈련/검증 데이터셋 분리 (80:20)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# RobustScaler를 사용하여 스케일링
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

---9

# Gradient Boosting은 여러 개의 약한 예측기(weak learners)를 순차적으로 결합하여 강력한 예측기를 만드는 앙상블(ensemble) 기법
# 각 단계에서 이전 예측기의 잔차(residual)를 보완하도록 새로운 예측기를 추가
# 이를 통해 모델의 성능을 점진적으로 향상시킬 수 있음

from sklearn.ensemble import GradientBoostingRegressor


# Gradient Boosting Regressor 모델 생성
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


# 훈련 데이터셋으로 모델 학습
gbr_model.fit(X_train_scaled, y_train)

---10

# XGBoost는 Gradient Boosting을 기반으로 하며, 병렬 처리와 효율적인 메모리 사용 등 최적화된 구현 으로 속도와 성능 면에서 뛰어남

import xgboost as xgb

# XGBoost Regressor 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 훈련 데이터셋으로 모델 학습
xgb_model.fit(X_train_scaled, y_train)

---11

# MAE(Mean Absolute Error)는 실제값과 예측값의 차이 절댓값에 대한 평균
# MAE는 오차의 크기를 직관적으로 나타내며, 이상치에 덜 민감하다는 장점

from sklearn.metrics import mean_absolute_error

# GradientBoostingRegressor 모델로 예측
gbr_pred = gbr_model.predict(X_valid_scaled)
# XGBoostRegressor 모델로 예측
xgb_pred = xgb_model.predict(X_valid_scaled)

# MAE 계산
gbr_mae = mean_absolute_error(y_valid, gbr_pred)
xgb_mae = mean_absolute_error(y_valid, xgb_pred)

print(f"GradientBoostingRegressor MAE: {gbr_mae:.4f}")
print(f"XGBoostRegressor MAE: {xgb_mae:.4f}")

# 더 낮은 MAE를 가진 모델 선택
if gbr_mae < xgb_mae:
    answer11 = 'GradientBoostingRegressor'
else:
    answer11 = 'XGBoostRegressor'

print(f"최종 선택 모델: {answer11}")


---12

# TensorFlow Keras는 딥러닝 모델을 쉽게 구축할 수 있는 API
# Sequential 모델은 층을 순차적으로 쌓아 올리는 구조에 적합
# Dense는 완전 연결(fully connected) 레이어로, activation='relu'는 비선형성을 추가
# Dropout 레이어는 과적합(overfitting)을 방지하기 위해 일부 뉴런을 무작위로 비활성화
# 회귀 문제이므로 출력 레이어의 활성화 함수는 linear를 사용
# model.compile()은 모델을 훈련에 사용할 수 있도록 설정하며, model.fit()은 실제로 모델을 학습

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 딥러닝 모델 생성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear') # 회귀 모델이므로 활성화 함수를 linear로 설정
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# 모델 학습
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_valid_scaled, y_valid)
)

--- 13

# history.history 객체에는 모델 학습 과정에서 에포크별로 기록된 손실(loss) 및 평가 지표(metrics) 값들이 딕셔너리 형태로 저장
# plt.plot() 함수를 사용하여 학습 MSE와 검증 MSE를 각각 그릴 수 있음
# 이 그래프를 통해 에포크가 진행됨에 따라 모델의 성능이 어떻게 변화하는지, 그리고 과적합이 발생하는지 여부를 시각적으로 확인가능
import matplotlib.pyplot as plt


# 학습 MSE와 검증 MSE 시각화

# figsize는 그림의 가로와 세로 길이를 인치 단위로 설정
plt.figure(figsize=(10, 6))

# history.history 딕셔너리에는 에포크(epoch)별 훈련 지표가 저장되어 있음
# label='Training MSE'는 이 선 그래프의 범례에 표시될 이름을 지정
plt.plot(history.history['mse'], label='Training MSE')


# history.history['val_mse']는 각 에포크에서의 검증 데이터에 대한 MSE 값을 포함
# label='Validation MSE'는 이 선 그래프의 범례에 표시될 이름을 지정
plt.plot(history.history['val_mse'], label='Validation MSE')

# 그래프의 제목을 'Model MSE over Epochs'로 설정
plt.title('Model MSE over Epochs')

# x축의 레이블을 'Epochs'로 설정
plt.xlabel('Epochs')

# y축의 레이블을 'MSE'로 설정
plt.ylabel('MSE')

# 그래프에 범례(legend)를 추가
# 범례는 각 선이 무엇을 의미하는지(예: Training MSE, Validation MSE)를 설명
plt.legend()

# 완성된 그래프를 화면에 표시
plt.show()


---14

# 11번 문항에서 계산된 값
# xgb_pred = xgb_model.predict(X_valid_scaled)
# xgb_mae = mean_absolute_error(y_valid, xgb_pred)

# model.predict()를 사용하여 딥러닝 모델의 예측값을 얻음
dl_pred = model.predict(X_valid_scaled)
dl_mae = mean_absolute_error(y_valid, dl_pred)


if xgb_mae < dl_mae:
    print(f"XGBoost 모델이 딥러닝 모델보다 성능이 우수합니다. 최종 MAE: {xgb_mae:.4f}")
else:
    print(f"딥러닝 모델이 XGBoost 모델보다 성능이 우수합니다. 최종 MAE: {dl_mae:.4f}")