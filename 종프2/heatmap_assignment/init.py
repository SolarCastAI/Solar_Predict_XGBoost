import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

# PyTorch 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 데이터 로드 및 전처리 (Data Loading & Preprocessing)
# ==========================================

def load_and_clean_data(road_path, weather_path):
    print(">>> 데이터 로딩 중...")
    
    # 1-1. 기상청 데이터 로드 (광주.csv)
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    weather_df = None
    
    for encoding in encodings:
        try:
            print(f">>> 기상 데이터 인코딩 시도: {encoding}")
            weather_df = pd.read_csv(weather_path, encoding=encoding)
            print(f">>> 기상 데이터 로드 성공 (인코딩: {encoding})")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f">>> {encoding} 시도 중 오류: {e}")
            continue
    
    if weather_df is None:
        raise ValueError("기상 데이터를 읽을 수 없습니다. 파일 인코딩을 확인해주세요.")
    
    print(f">>> 기상 데이터 컬럼: {weather_df.columns.tolist()}")
    print(f">>> 기상 데이터 날짜 범위: {weather_df['발전일자'].min()} ~ {weather_df['발전일자'].max()}")
    
    # 필요한 컬럼 선택 및 이름 변경
    weather_df = weather_df[['발전일자', '기온', '습도', '풍속', '일사량']]
    weather_df.columns = ['datetime', 'air_temp', 'humidity', 'wind_speed', 'insolation']
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df = weather_df.set_index('datetime').sort_index()
    
    # 결측치 처리 (선형 보간)
    weather_df = weather_df.interpolate(method='time').fillna(0)

    # 1-2. 도로/IoT 데이터 로드 (양산교차로.csv)
    road_df = None
    
    for encoding in encodings:
        try:
            print(f">>> 도로 데이터 인코딩 시도: {encoding}")
            road_df = pd.read_csv(road_path, encoding=encoding)
            print(f">>> 도로 데이터 로드 성공 (인코딩: {encoding})")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f">>> {encoding} 시도 중 오류: {e}")
            continue
    
    if road_df is None:
        raise ValueError("도로 데이터를 읽을 수 없습니다. 파일 인코딩을 확인해주세요.")
    
    print(f">>> 도로 데이터 shape: {road_df.shape}")
    print(f">>> 도로 데이터 컬럼: {road_df.columns.tolist()}")
    
    # 도로 데이터 재구조화 (여러 지점이 옆으로 나열된 구조 → 세로로 변환)
    # 패턴: 일시, 기온(°C), 습도, 지점명 | 일시.1, 기온(°C).1, 습도.1, 지점명.1 ...
    
    # 마지막 센서 데이터 추출 (일시.3, 기온(°C).3, 지점명.3)
    # 습도가 없는 경우도 있으므로 유연하게 처리
    datetime_cols = [col for col in road_df.columns if '일시' in col]
    temp_cols = [col for col in road_df.columns if '기온' in col]
    
    print(f">>> 발견된 날짜 컬럼: {datetime_cols}")
    print(f">>> 발견된 온도 컬럼: {temp_cols}")
    
    # 마지막 지점 데이터 선택 (인덱스 기준)
    if len(datetime_cols) > 0 and len(temp_cols) > 0:
        target_datetime_col = datetime_cols[-1]  # 마지막 일시 컬럼
        target_temp_col = temp_cols[-1]  # 마지막 기온 컬럼
        
        target_data = road_df[[target_datetime_col, target_temp_col]].copy()
        target_data.columns = ['datetime', 'surface_temp']
        
        target_data['datetime'] = pd.to_datetime(target_data['datetime'], errors='coerce')
        target_data = target_data.dropna(subset=['datetime'])
        target_data['surface_temp'] = pd.to_numeric(target_data['surface_temp'], errors='coerce')
        
        print(f">>> 도로 데이터 날짜 범위: {target_data['datetime'].min()} ~ {target_data['datetime'].max()}")
        
        # 시간 단위 통합 (1시간 단위로 Resampling)
        target_data = target_data.set_index('datetime').resample('1h').mean()
        
    else:
        raise ValueError("도로 데이터에서 날짜 또는 온도 컬럼을 찾을 수 없습니다.")
    
    # 1-3. 데이터 병합 전 날짜 범위 확인
    print(f">>> 기상 데이터 인덱스 범위: {weather_df.index.min()} ~ {weather_df.index.max()}")
    print(f">>> 도로 데이터 인덱스 범위: {target_data.index.min()} ~ {target_data.index.max()}")
    
    # 1-4. 데이터 병합 (Merge)
    merged_df = pd.merge(weather_df, target_data, left_index=True, right_index=True, how='inner')
    
    print(f">>> 병합된 데이터 크기: {merged_df.shape}")
    
    if merged_df.empty:
        print("=" * 60)
        print("⚠️  경고: 두 데이터셋의 날짜 범위가 겹치지 않습니다!")
        print(f"기상 데이터: {weather_df.index.min()} ~ {weather_df.index.max()}")
        print(f"도로 데이터: {target_data.index.min()} ~ {target_data.index.max()}")
        print("=" * 60)
        
        # 날짜가 안 맞는 경우 대안: 도로 데이터 날짜를 기상 데이터 범위로 매핑
        print(">>> 대안: 도로 데이터의 시간을 기상 데이터 범위에 맞춰 조정합니다...")
        
        # 도로 데이터의 시간대만 추출하여 기상 데이터 날짜에 매핑
        target_data_reset = target_data.reset_index()
        target_data_reset['time_only'] = target_data_reset['datetime'].dt.time
        
        # 기상 데이터에서 동일한 시간대 찾기
        weather_df_reset = weather_df.reset_index()
        weather_df_reset['time_only'] = weather_df_reset['datetime'].dt.time
        
        # 시간대 기준으로 병합 (날짜 무시)
        merged_df = pd.merge(
            weather_df_reset, 
            target_data_reset[['time_only', 'surface_temp']], 
            on='time_only', 
            how='inner'
        )
        merged_df = merged_df.drop(columns=['time_only']).set_index('datetime')
        
        print(f">>> 시간대 기준 병합 후 데이터 크기: {merged_df.shape}")
    
    print(f">>> 병합된 데이터 샘플:\n{merged_df.head()}")
    print(f">>> 결측치 확인:\n{merged_df.isnull().sum()}")
    
    # 결측치 제거
    merged_df = merged_df.dropna()
    
    return merged_df

# ==========================================
# 2. K-Means 클러스터링 (Pattern Analysis)
# ==========================================

def add_weather_clusters(df, n_clusters=4):
    print(">>> K-Means 클러스터링 수행 중...")
    
    # 클러스터링에 사용할 기상 특징 선택
    features = df[['air_temp', 'humidity', 'insolation', 'wind_speed']]
    
    # 스케일링
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-Means 학습
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['weather_cluster'] = kmeans.fit_predict(features_scaled)
    
    # 클러스터 결과를 One-Hot Encoding
    df = pd.get_dummies(df, columns=['weather_cluster'], prefix='cluster')
    
    return df

# ==========================================
# 3. 데이터셋 생성 (Windowing for LSTM)
# ==========================================

def create_sequences(data, target, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# PyTorch Dataset 클래스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 4. Attention LSTM/GRU 모델 정의 (PyTorch)
# ==========================================

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, lstm_output):
        attention_weights = torch.softmax(
            torch.bmm(lstm_output, lstm_output.transpose(1, 2)), 
            dim=-1
        )
        context = torch.bmm(attention_weights, lstm_output)
        return context

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(AttentionRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(lstm_out)
        attention_out = self.attention(gru_out)
        combined = torch.cat([gru_out, attention_out], dim=-1)
        last_output = combined[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ==========================================
# 5. 학습 함수
# ==========================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# ==========================================
# 6. 예측 함수
# ==========================================

def predict(model, data_loader, device='cpu'):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

# ==========================================
# 7. 실행 및 앙상블 (Main Pipeline)
# ==========================================

# 경로 설정
road_file_path = './dataset/양산교차로.csv'
weather_file_path = './dataset/output_by_region/광주.csv'

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 사용 디바이스: {device}")

# 1. 데이터 로드
try:
    df = load_and_clean_data(road_file_path, weather_file_path)
    
    if df.empty or len(df) < 50:
        raise ValueError(f"데이터가 부족합니다. 현재 데이터 개수: {len(df)}")

    # 2. 패턴 분석 (Clustering)
    df = add_weather_clusters(df)

    # 3. 학습용 데이터 준비
    target_col = 'surface_temp'
    feature_cols = [c for c in df.columns if c != target_col]
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_x = scaler_x.fit_transform(df[feature_cols])
    scaled_y = scaler_y.fit_transform(df[[target_col]])
    
    TIME_STEPS = 24
    X, y = create_sequences(scaled_x, scaled_y, TIME_STEPS)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 딥러닝 모델 학습
    print(">>> 딥러닝 모델 학습 시작 (LSTM/GRU + Attention)...")
    
    input_size = X_train.shape[2]
    model = AttentionRNN(input_size=input_size, hidden_size=64, dropout=0.2).to(device)
    
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        epochs=50, lr=0.001, device=device
    )
    
    dl_pred_train = predict(model, train_loader, device)
    dl_pred_test = predict(model, test_loader, device)
    
    # 5. Stacking Ensemble (XGBoost)
    print(">>> Stacking Ensemble (XGBoost) 학습 시작...")
    
    X_train_meta = np.concatenate([dl_pred_train, X_train[:, -1, :]], axis=1)
    X_test_meta = np.concatenate([dl_pred_test, X_test[:, -1, :]], axis=1)
    
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    xgb_model.fit(X_train_meta, y_train)
    
    final_pred = xgb_model.predict(X_test_meta)
    
    final_pred_inv = scaler_y.inverse_transform(final_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test)
    
    # 6. 최종 성능 평가
    nmae = mean_absolute_error(y_test_inv, final_pred_inv) / np.mean(y_test_inv)
    r2 = r2_score(y_test_inv, final_pred_inv)
    
    print("="*40)
    print(f"최종 성능 평가 결과:")
    print(f"R2 Score (결정계수): {r2:.4f}")
    print(f"NMAE (정규화 평균 절대 오차): {nmae:.4f}")
    print("="*40)
    
    # 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv[:100], label='Actual Road Temp', color='blue')
    plt.plot(final_pred_inv[:100], label='Predicted Road Temp (Stacked)', color='red', linestyle='--')
    plt.title('Road Heat Prediction: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./heatmap_prediction/prediction_result.png', dpi=300)
    plt.show()
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./heatmap_prediction/training_loss.png', dpi=300)
    plt.show()

except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()