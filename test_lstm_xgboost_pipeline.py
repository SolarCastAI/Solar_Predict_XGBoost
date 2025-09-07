import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost import plot_importance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-whitegrid')


def calculate_rmse(y_true, y_pred):
    """RMSE를 계산합니다."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """MAPE를 계산합니다."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 0으로 나누는 것을 방지하기 위해 y_true가 0인 경우는 제외
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100



def xgb_model(X_train, y_train, X_val, y_val, plotting=False):
    """Pre-optimized XGBoost model을 학습시키고 MAE와 모델을 반환합니다."""
    # XGBoost 모델 정의
    xgb_regressor = xgb.XGBRegressor(
        gamma=1, 
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        early_stopping_rounds=50 
    )
    
    # 모델 학습
    xgb_regressor.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False # 학습 과정을 출력하지 않음
    )
    
    # 검증 데이터에 대한 예측
    pred_val = xgb_regressor.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    
    # 결과 시각화
    if plotting:
        plt.figure(figsize=(15, 6))
        sns.lineplot(x=range(len(y_val)), y=y_val.values, color="grey", alpha=.4, label="Actual")
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red", label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("DC Power")
        plt.title(f"XGBoost Validation - MAE: {round(mae, 3)}")
        plt.legend()
        plt.show()
        
    return mae, xgb_regressor

#=================================
# LSTM 모델을 위한 데이터셋 생성 함수
#=================================
def create_sequences(X, y, time_steps=1):
    """LSTM 모델 입력을 위한 시퀀스 데이터를 생성합니다."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


#=================================
# LSTM 모델 함수
#=================================
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

def lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, time_steps, n_features, EPOCH=50, BATCH_SIZE=64):
    """LSTM 모델을 정의, 컴파일, 학습하고 모델을 반환합니다."""
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get("val_mae") < 150: # 목표 MAE 도달 시 조기 종료 (스케일된 값이므로 실제 값에 맞게 조정 필요)
                print("\nValidation MAE is low, so stopping training!")
                self.model.stop_training = True

    callbacks = [
        # myCallback(), # 스케일링된 mae는 해석이 어려우므로 EarlyStopping만 사용
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    model = Sequential([
        LSTM(64, input_shape=(time_steps, n_features), return_sequences=True),
        LSTM(32, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
    
    history = model.fit(
        X_train_seq, y_train_seq,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        callbacks=callbacks,
        validation_data=(X_val_seq, y_val_seq),
        verbose=1
    )
    
    return model, history

# --- 1. 데이터 로드 ---
try:
    gen_df = pd.read_csv(r'./dataset/Plant_1_Generation_Data.csv')
    weather_df = pd.read_csv(r'./dataset/Plant_1_Weather_Sensor_Data.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. 파일 경로를 다시 확인해주세요.")
    exit()


# --- 2. 데이터 전처리 및 병합 ---
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

gen_agg_df = gen_df.groupby('DATE_TIME').agg({
    'DC_POWER': 'sum',
    'AC_POWER': 'sum',
    'DAILY_YIELD': 'sum'
}).reset_index()

df = pd.merge(weather_df, gen_agg_df, on='DATE_TIME', how='inner')
df = df.drop(['PLANT_ID', 'SOURCE_KEY'], axis=1)
df.set_index('DATE_TIME', inplace=True)


# --- 3. 피처 엔지니어링 ---
df['HOUR'] = df.index.hour
df['DAY_OF_WEEK'] = df.index.dayofweek
df['DAY_OF_YEAR'] = df.index.dayofyear
df['MONTH'] = df.index.month

print("데이터 전처리 및 피처 엔지니어링 완료. 최종 데이터 형태:")
print(df.head())


# --- 4. 데이터 분할 (시계열 특성 유지) ---
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR', 'DAY_OF_WEEK', 'DAY_OF_YEAR', 'MONTH']
target = 'DC_POWER'

X = df[features]
y = df[target]

train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

# 훈련 데이터 (70%)
X_train, y_train = X[:train_size], y[:train_size]

# 검증 데이터 (15%) - Typo corrected here
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]

# 테스트 데이터 (15%)
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]


# --- 5. 데이터 스케일링 (LSTM에 사용) ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=features, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler_X.transform(X_val), columns=features, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), columns=features, index=X_test.index)

y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
y_val_scaled = pd.Series(scaler_y.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index)
y_test_scaled = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)

print("\n데이터 스케일링 완료.")


# =================================================================
# === 스태킹 앙상블 모델링 파트 ===
# =================================================================

# --- 6. LSTM (Base-Model) 학습 및 예측 ---
print("\n--- LSTM (Base-Model) 학습 및 예측 시작 ---")
TIME_STEPS = 24 
N_FEATURES = len(features)

# LSTM 학습을 위한 시퀀스 데이터 생성
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, TIME_STEPS)
X_test_seq, _ = create_sequences(X_test_scaled, y_test_scaled, TIME_STEPS) # y는 예측에 불필요

# LSTM 모델 학습
model_lstm, history_lstm = lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, TIME_STEPS, N_FEATURES)

# 전체 데이터셋에 대해 LSTM 예측값 생성 (XGBoost의 새로운 피처로 사용)
pred_lstm_train_scaled = model_lstm.predict(X_train_seq)
pred_lstm_val_scaled = model_lstm.predict(X_val_seq)
pred_lstm_test_scaled = model_lstm.predict(X_test_seq)

# 예측 결과를 원래 스케일로 복원
pred_lstm_train = scaler_y.inverse_transform(pred_lstm_train_scaled).flatten()
pred_lstm_val = scaler_y.inverse_transform(pred_lstm_val_scaled).flatten()
pred_lstm_test = scaler_y.inverse_transform(pred_lstm_test_scaled).flatten()

print("LSTM 예측 완료 및 스케일 복원 완료.")


# --- 7. XGBoost (Meta-Model)를 위한 데이터 준비 ---
print("\n--- XGBoost (Meta-Model)를 위한 데이터 준비 ---")

# 시퀀스 생성으로 인해 길이가 줄어든 y 데이터와 X 데이터를 정렬
y_train_aligned = y_train.iloc[TIME_STEPS:]
y_val_aligned = y_val.iloc[TIME_STEPS:]
y_test_aligned = y_test.iloc[TIME_STEPS:]

X_train_aligned = X_train.iloc[TIME_STEPS:]
X_val_aligned = X_val.iloc[TIME_STEPS:]
X_test_aligned = X_test.iloc[TIME_STEPS:]

# LSTM 예측값을 새로운 피처로 추가
X_train_stacked = X_train_aligned.copy()
X_train_stacked['LSTM_PREDICTION'] = pred_lstm_train

X_val_stacked = X_val_aligned.copy()
X_val_stacked['LSTM_PREDICTION'] = pred_lstm_val

X_test_stacked = X_test_aligned.copy()
X_test_stacked['LSTM_PREDICTION'] = pred_lstm_test

print("스태킹을 위한 새로운 피처셋 생성 완료:")
print(X_train_stacked.head())


# --- 8. XGBoost (Meta-Model) 학습 및 최종 예측 ---
print("\n--- XGBoost (Meta-Model) 학습 시작 ---")

mae_stacked_val, model_stacked = xgb_model(
    X_train_stacked, y_train_aligned, 
    X_val_stacked, y_val_aligned, 
    plotting=False # 여기서는 최종 결과만 볼 것이므로 중간 시각화는 생략
)
print(f"Stacked Model Validation MAE: {mae_stacked_val:.4f}")

# 테스트 데이터로 최종 예측
pred_stacked = model_stacked.predict(X_test_stacked)


# --- 9. 최종 성능 평가 ---
print("\n--- 최종 스태킹 앙상블 모델 성능 평가 ---")

mae_stacked = mean_absolute_error(y_test_aligned, pred_stacked)
rmse_stacked = calculate_rmse(y_test_aligned, pred_stacked)
mape_stacked = calculate_mape(y_test_aligned, pred_stacked)

results = {
    "Metric": ["MAE", "RMSE", "MAPE (%)"],
    "Stacked Ensemble (LSTM + XGBoost)": [mae_stacked, rmse_stacked, mape_stacked]
}
results_df = pd.DataFrame(results).set_index("Metric")
print(results_df)


# --- 10. 결과 시각화 ---
plt.figure(figsize=(20, 8))

plt.plot(y_test_aligned.index, y_test_aligned, label='Actual DC Power', color='blue', alpha=0.6)
plt.plot(y_test_aligned.index, pred_stacked, label=f'Stacked Ensemble Prediction (MAE: {mae_stacked:.2f})', color='purple', linestyle='--')

plt.title('Solar Power Generation Prediction using Stacked Ensemble Model', fontsize=16)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('DC Power (kW)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 스태킹 모델의 피처 중요도 시각화
fig, ax = plt.subplots(figsize=(12, 6))
plot_importance(model_stacked, ax=ax, importance_type='gain')
plt.title('Stacked XGBoost Feature Importance (Gain)')
plt.show()