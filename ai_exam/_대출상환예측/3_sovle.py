################1

# 모델링 전체 파이프라인에 필요한 핵심 라이브러리를 불러오고, 서로 다른 포맷(JSON/CSV)의 원천 데이터를 DataFrame으로 로딩

import pandas as pd
import sklearn as sk


# 신청자 기본 정보 로딩: JSON은 컬럼형 구조(Records)라고 가정
df_applicants = pd.read_json('loan_applicants.json')

# 금융 이력 로딩: CSV는 기본 구분자(,) 사용, 헤더 자동 인식
df_history = pd.read_csv('financial_history.csv')

#1 my slove
import pandas as pd
import sklearn as sk

df_applicants = pd.read_json('loan_applicants.json')
df_history = pd.read_csv('financial_history.csv')


###2

 # 두 데이터 소스를 ApplicantID 기준으로 결합하여 학습에 사용할 통합 테이블을 만듦
# inner join을 사용하여 양쪽 모두에 존재하는 신청자만 남김(결측/단절 레코드 제거 효과)
df = pd.merge(df_applicants, df_history, on='ApplicantID', how='inner')

# 구조 점검: 각 컬럼의 dtype, 결측치 존재 여부, 메모리 사용량을 확인해 전처리 계획을 세움
df.info()

# 내용 스냅샷: 수치 범위(예: CreditScore, Income), 범주 표기 일관성(대소문, 스페이스) 등을 육안 확인
df.head()

###2 my solve
df = pd.merge(df_applicants, df_history, how='inner', on='ApplicantID')

df.info()
df.head()


###3
# 범주형 컬럼(EmploymentType)의 분포를 시각화하여 클래스 불균형/이상 범주 여부를 파악
import matplotlib.pyplot as plt
import seaborn as sns


# (정제 전) 분포 시각화: 소수 클래스가 과도하면 모델이 편향될 수 있다.
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='EmploymentType')
plt.title('EmploymentType Distribution (Before Cleaning)')
plt.xlabel('EmploymentType'); plt.ylabel('Count')
plt.tight_layout(); plt.show()


# 행 삭제 및 건수 계산
initial_rows = df.shape[0]
df = df[df['EmploymentType'] != 'Unemployed'].copy()

답안03 = initial_rows - df.shape[0]


# (정제 후) 분포 재시각화: 제거 효과 확인(클래스 비율 변화)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='EmploymentType')
plt.title('EmploymentType Distribution (After Cleaning)')
plt.xlabel('EmploymentType'); plt.ylabel('Count')
plt.tight_layout(); plt.show()

### 3 my_solve

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 8))
sns.countplot(data=df, x='EmploymentType')
plt.show()

df_del_unem = df[df['EmploymentType'] != 'Unemployed']

before_del = df.shape[0]
after_del = df_del_unem.shape[0]

답안03 = before_del - after_del
print(답안03)


### 4

# 소득과 상환점수 간의 관계를 이변량으로 확인하고, 극단치가 존재하는지 빠르게 스크리닝

sns.jointplot(data=df, x='Income_USD', y='RepaymentScore')
plt.suptitle('Income vs RepaymentScore', y=1.02)
plt.show()

# 이상치 개수 산정: 규칙 기반 필터로 개수만 취합(후속 단계에서 제거)
답안04 = df[df['Income_USD'] >= 300000].shape[0]

### 4 my solve
plt.figure(figsize=(8,4))
sns.jointplot(data = df, x = 'Income_USD')
plt.show()

답안04 = df[df['Income_USD'] >= 300000]
print(답안04)

### 5

# 이상치를 제거해 회귀 계수/손실에 대한 영향(스케일 왜곡)을 줄이고, 예측에 불필요한 고유 식별자(ApplicantID)를 제거해 데이터 누수를 방지

df_temp = df[df['Income_USD'] < 300000].copy()
df_temp.drop('ApplicantID', axis=1, inplace=True)

### 5. my_solve

df_temp = df[df['Income_USD'] < 300000]
print(df_temp)

df_temp.drop(['ApplicantID'], axis=1)

print(df_temp)


### 6

# 결측치를 평균값으로 대체하여 데이터의 손실을 최소화하고, 결측 데이터로 인한 모델 성능 저하를 방지합니다.
# 평균 대체는 수치형 데이터에 주로 사용되는 방법입니다.

# 각 컬럼의 결측 개수를 확인합니다.
print(df_temp.isnull().sum())

# 'AnnualIncome' 컬럼의 평균을 계산하고, 해당 컬럼의 결측치를 평균값으로 대체합니다.
mean_income = df_temp['Income_USD'].mean()
df_temp['Income_USD'].fillna(mean_income, inplace=True)

# 'CreditScore' 컬럼의 평균을 계산하고, 해당 컬럼의 결측치를 평균값으로 대체합니다.
mean_score = df_temp['CreditScore'].mean()
df_temp['CreditScore'].fillna(mean_score, inplace=True)

# 결측치가 처리된 데이터를 df_na에 저장합니다.
df_na = df_temp.copy()

# 대체된 총 결측치 개수를 계산합니다.
# 원래의 결측치 개수를 다시 확인하여 합산합니다.
# (예: df_temp.isnull().sum()를 다시 실행하여 합산)
답안06 = df_temp['Income_USD'].isnull().sum() + df_temp['CreditScore'].isnull().sum()

print("대체된 결측치 개수:", 답안06)

### 7

# 목적: 범주형 변수를 get_dummies로 원-핫 인코딩하여 모델이 해석 가능한 수치 피처로 변환
# 선택: drop_first=True로 기준 범주를 제거하여 다중공선성을 완화
df_encoded = pd.get_dummies(df_na, columns=['EmploymentType','EducationLevel'], drop_first=True)

# 변환 결과 확인(샘플)
print(df_encoded.head())


### 8

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 타깃/피처 분리
X = df_encoded.drop('RepaymentScore', axis=1)
y = df_encoded['RepaymentScore']

# 데이터 분할(재현성 고정)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 표준화: 훈련 세트로만 적합(fit)하고 검증 세트엔 변환(transform)만 적용해 누수 방지
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

### 8 my_solve

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df_encoded.drop('RepaymentScore', axis=1)
y = df_encoded['RepaymentScore']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_valid_scaled = scalar.transform(X_valid)



### 9
from sklearn.linear_model import LinearRegression

linreg_model = LinearRegression()              # 정규화 없는 OLS 추정
linreg_model.fit(X_train_scaled, y_train)      # 최소제곱으로 가중치 학습


### 10

from xgboost import XGBRegressor

xgb_model = XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
xgb_model.fit(X_train_scaled, y_train)

### 11

# 모델 성능을 정량적으로 평가하기 위해 MAE, RMSE, R²를 계산한다.
# MAE는 절대 오차 평균으로 해석이 직관적이고, RMSE는 큰 오차에 더 큰 패널티를 준다.
# R²는 설명력(variance 설명 비율)을 보여준다.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 선형 회귀 예측 및 성능 평가
y_pred_lin = linreg_model.predict(X_valid_scaled)
mae_lin = mean_absolute_error(y_valid, y_pred_lin)
rmse_lin = np.sqrt(mean_squared_error(y_valid, y_pred_lin))
r2_lin = r2_score(y_valid, y_pred_lin)

# XGBoost 예측 및 성능 평가
y_pred_xgb = xgb_model.predict(X_valid_scaled)
mae_xgb = mean_absolute_error(y_valid, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))
r2_xgb = r2_score(y_valid, y_pred_xgb)

print("Linear Regression - MAE:", mae_lin, "RMSE:", rmse_lin, "R²:", r2_lin)
print("XGBoost Regressor - MAE:", mae_xgb, "RMSE:", rmse_xgb, "R²:", r2_xgb)


### 11 my_solve

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

#
y_lin_pred = linreg_model.predict(X_valid_scaled)
lin_rmse = np.sqrt(mean_squared_error(y_valid, y_lin_pred))
lin_mae = mean_absolute_error(y_valid, y_lin_pred)
lin_r2 = r2_score(y_valid, y_lin_pred)
#
y_xgb_pred = xgb_model.predict(X_valid_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_valid, y_xgb_pred))
xgb_mae = mean_absolute_error(y_valid, y_xgb_pred)
xgb_r2 = r2_score(y_valid, y_xgb_pred)

print('lin: ', y_lin_pred, lin_rmse, lin_mae, lin_r2)
print('xgb: ', y_xgb_pred, xgb_rmse, xgb_mae, xgb_r2)


### 12

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

tf.random.set_seed(42)

input_dim = X_train_scaled.shape[1]

dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),  # 첫 번째 은닉층
    Dropout(0.2),                                           # 과적합 방지
    Dense(32, activation='relu'),                           # 두 번째 은닉층
    Dense(1)                                                # 출력층 (회귀)
])

dl_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

history_dl = dl_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    epochs=20,
    batch_size=16,
    verbose=0
)

### 12 my_solve

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

input_dim = X_train_scaled.shape[1]

dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

dl_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = dl_model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=20, batch_size=16)


### 13

# 목적: 학습 과정에서 과적합 여부 및 수렴 패턴을 시각적으로 확인한다.
train_mse = history_dl.history['mse']
val_mse = history_dl.history['val_mse']
epochs = range(1, len(train_mse) + 1)

plt.figure(figsize=(8,5))

plt.plot(epochs, train_mse, label='Training MSE')
plt.plot(epochs, val_mse, label='Validation MSE')
plt.title('DNN Model Training vs Validation MSE')

plt.xlabel('Epochs')
plt.ylabel('MSE')

plt.legend()
plt.grid(True)
plt.show()


### 13 my_solve

import matplotlib.pyplot as plt
import seaborn as sns

history_dl_mse = history_dl.history['mse']
history_val_dl_mse = history_dl.history['val_mse']
epochs = range(0, len(history_dl_mse) + 1)

plt.figure(figsize=(8, 4))
plt.plot(history_dl_mse)
plt.plot(history_val_dl_mse)
plt.show()


### 14.

# 목적: 세 가지 모델의 예측 성능을 비교하고 최종적으로 실무 적용 가능성이 높은 모델을 선택한다.

# DNN 모델 예측 및 MAE 계산
y_pred_dl = dl_model.predict(X_valid_scaled).flatten()
mae_dl = mean_absolute_error(y_valid, y_pred_dl)

print("Linear Regression MAE:", mae_lin)
print("XGBoost MAE:", mae_xgb)
print("DNN MAE:", mae_dl)

# 최소 MAE 모델 선택
min_mae = min(mae_lin, mae_xgb, mae_dl)
if min_mae == mae_lin:
    답안14 = 'Linear Regression'
elif min_mae == mae_xgb:
    답안14 = 'XGBoost Regressor'
else:
    답안14 = 'Deep Learning (DNN)'

print("최종 선택 모델 (답안14):", 답안14)

# 선택 근거 예시:
# - 선형 회귀: 해석 용이, 구현 간단, 데이터가 단순 선형 관계일 때 적합
# - XGBoost: 높은 성능과 안정성, 피처 중요도 확인 가능
# - DNN: 비선형 패턴 학습 가능, 다만 해석과 배포 복잡도 존재