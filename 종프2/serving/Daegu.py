import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import pickle
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

# GPU/CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 사용 중인 디바이스: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    print("   ⚠️  CUDA를 사용할 수 없습니다. CPU를 사용합니다.")

warnings.filterwarnings("ignore")

# 디렉토리 설정
output_dir = "./plots_daegu_transfer"
model_dir = "./saved_models"  # 제주 백본 모델이 저장된 디렉토리

os.makedirs(output_dir, exist_ok=True)

print(f"📁 Plot 저장 경로: {output_dir}")
print(f"📁 제주 백본 모델 경로: {model_dir}")

plt.style.use('seaborn-v0_8-whitegrid')

# 한글 폰트 설정
import matplotlib.font_manager as fm
import platform

def set_korean_font():
    system = platform.system()
    korean_fonts = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 
                   'NanumBarunGothic', 'Nanum Gothic', 'DejaVu Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"✅ 한글 폰트 설정 완료: {font}")
            break
    else:
        print("⚠️  한글 폰트를 찾을 수 없습니다.")
    
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


# === 평가 지표 함수들 ===
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_mape(y_true, y_pred, method='improved'):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    if method == 'improved':
        threshold = np.percentile(y_true_clean, 10)
        significant_mask = y_true_clean >= threshold
        
        if not np.any(significant_mask):
            abs_errors = np.abs(y_true_clean - y_pred_clean)
            mean_actual = np.mean(y_true_clean)
            if mean_actual > 0:
                return (np.mean(abs_errors) / mean_actual) * 100
            else:
                return 0.0
        
        y_true_sig = y_true_clean[significant_mask]
        y_pred_sig = y_pred_clean[significant_mask]
        
        weights = y_true_sig / np.sum(y_true_sig)
        percentage_errors = np.abs((y_true_sig - y_pred_sig) / y_true_sig)
        percentage_errors = np.clip(percentage_errors, 0, 2)
        
        mape_value = np.sum(weights * percentage_errors) * 100
        
    return mape_value

def calculate_all_metrics(y_true, y_pred, print_details=False):
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred, method='improved'),
    }
    
    data_range = np.max(y_true) - np.min(y_true)
    metrics['nmae'] = metrics['mae'] / data_range if data_range > 0 else 0
    metrics['nrmse'] = metrics['rmse'] / data_range if data_range > 0 else 0
    
    if print_details:
        print(f"\n=== 평가 지표 ===")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"NMAE: {metrics['nmae']:.4f}")
        print(f"NRMSE: {metrics['nrmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
    
    return metrics


# load_daegu_data 함수 내에서 feature_cols 부분을 수정
def load_daegu_data(file_path, sequence_length=24):
    """
    대구 CSV 데이터 로딩 및 전처리
    """
    print("\n" + "="*80)
    print("대구 데이터 로딩 중...")
    print("="*80)
    
    df = pd.read_csv(file_path)
    print(f"원본 데이터 크기: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    
    # 날짜 처리
    df['발전일자'] = pd.to_datetime(df['발전일자'])
    
    # 컬럼 매핑
    column_mapping = {
        '발전일자': 'datetime',
        '기온': 'temperature',
        '강우량(mm)': 'precipitation',
        '습도': 'humidity',
        '적설량(mm)': 'snow',
        '적운량(10분위)': 'cloud_cover',
        '일조(hr)': 'sunshine_duration',
        '일사량': 'solar_radiation',
        '설비용량(MW)': 'solar_capacity',
        '발전량(MWh)': 'solar_generation'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    
    # 결측치 처리
    df_renamed['precipitation'] = df_renamed['precipitation'].fillna(0)
    df_renamed['snow'] = df_renamed['snow'].fillna(0)
    df_renamed['sunshine_duration'] = df_renamed['sunshine_duration'].fillna(0)
    df_renamed['solar_radiation'] = df_renamed['solar_radiation'].fillna(0)
    df_renamed['humidity'] = df_renamed['humidity'].fillna(df_renamed['humidity'].mean())
    df_renamed['temperature'] = df_renamed['temperature'].fillna(df_renamed['temperature'].mean())
    df_renamed['cloud_cover'] = df_renamed['cloud_cover'].fillna(5)
    
    # 시간 특성 추가
    df_renamed['hour'] = df_renamed['datetime'].dt.hour
    df_renamed['month'] = df_renamed['datetime'].dt.month
    df_renamed['day_of_year'] = df_renamed['datetime'].dt.dayofyear
    df_renamed['is_daytime'] = ((df_renamed['hour'] >= 6) & (df_renamed['hour'] <= 18)).astype(int)
    
    # 태양 고도각 (대구 위도 35.87)
    latitude = 35.87
    df_renamed['solar_altitude'] = np.sin(np.radians(
        90 - latitude + 23.45 * np.sin(np.radians(360/365 * (df_renamed['day_of_year'] - 81)))
    )) * np.sin(np.radians(15 * (df_renamed['hour'] - 12)))
    
    print(f"\n데이터 기간: {df_renamed['datetime'].min()} ~ {df_renamed['datetime'].max()}")
    print(f"평균 발전량: {df_renamed['solar_generation'].mean():.2f} MWh")
    print(f"설비용량: {df_renamed['solar_capacity'].iloc[0]:.2f} MW")
    
    # ⭐ 제주 모델과 동일한 8개 특성만 선택
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'cloud_cover',
        'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour'
    ]
    
    print(f"\n⚠️  제주 모델 호환을 위해 {len(feature_cols)}개 특성 사용:")
    print(f"   {feature_cols}")
    
    target_col = 'solar_generation'
    
    # 유효한 데이터만 선택
    df_valid = df_renamed[df_renamed[target_col].notna()].copy()
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values.reshape(-1, 1)
    dates = df_valid['datetime'].values
    
    # 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 시퀀스 데이터 생성
    X_seq, y_seq, date_seq = [], [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
        date_seq.append(dates[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"\n시퀀스 데이터 shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Train/Val/Test 분할 (80/10/10)
    X_temp, X_test, y_temp, y_test, date_temp, date_test = train_test_split(
        X_seq, y_seq, date_seq, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val, date_train, date_val = train_test_split(
        X_temp, y_temp, date_temp, test_size=0.111, random_state=42
    )
    
    print(f"\n데이터 분할:")
    print(f"  Train: {X_train.shape} ({len(X_train)/len(X_seq)*100:.1f}%)")
    print(f"  Val: {X_val.shape} ({len(X_val)/len(X_seq)*100:.1f}%)")
    print(f"  Test: {X_test.shape} ({len(X_test)/len(X_seq)*100:.1f}%)")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            scaler_X, scaler_y, feature_cols, df_valid, date_test)


# === PyTorch Dataset ===
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# === LSTM 모델 ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output


# === GRU 모델 ===
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out[:, -1, :])
        output = self.fc(gru_out)
        return output


# === 제주 모델 로드 함수 ===
def load_jeju_pretrained_models(model_dir='./saved_models', timestamp=None):
    """
    제주 데이터로 사전학습된 모델 로드
    """
    print(f"\n{'='*80}")
    print("제주 사전학습 모델 로드 중...")
    print(f"{'='*80}")
    
    # 최신 모델 정보 로드
    if timestamp is None:
        latest_path = os.path.join(model_dir, 'latest_models.json')
        if not os.path.exists(latest_path):
            raise FileNotFoundError(f"최신 모델 정보 파일을 찾을 수 없습니다: {latest_path}")
        
        with open(latest_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        timestamp = model_info['timestamp']
        print(f"최신 모델 타임스탬프: {timestamp}")
    
    # 메타데이터 로드
    metadata_path = os.path.join(model_dir, f'metadata_{timestamp}.json')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✅ 메타데이터 로드: {metadata_path}")
    
    # LSTM 모델 로드
    lstm_path = os.path.join(model_dir, f'lstm_model_{timestamp}.pth')
    lstm_checkpoint = torch.load(lstm_path, map_location=device)
    lstm_config = lstm_checkpoint['model_config']
    
    lstm_model = LSTMModel(
        input_size=lstm_config['input_size'],
        hidden_size=lstm_config['hidden_size'],
        num_layers=lstm_config['num_layers']
    ).to(device)
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    print(f"✅ LSTM 사전학습 모델 로드: {lstm_path}")
    
    # GRU 모델 로드
    gru_path = os.path.join(model_dir, f'gru_model_{timestamp}.pth')
    gru_checkpoint = torch.load(gru_path, map_location=device)
    gru_config = gru_checkpoint['model_config']
    
    gru_model = GRUModel(
        input_size=gru_config['input_size'],
        hidden_size=gru_config['hidden_size'],
        num_layers=gru_config['num_layers']
    ).to(device)
    gru_model.load_state_dict(gru_checkpoint['model_state_dict'])
    print(f"✅ GRU 사전학습 모델 로드: {gru_path}")
    
    print(f"\n{'='*80}")
    print(f"✨ 제주 사전학습 모델 로드 완료!")
    print(f"{'='*80}")
    
    return lstm_model, gru_model, metadata


# === 전이학습 함수 ===
def transfer_learning(model, train_loader, val_loader, criterion, 
                     num_epochs=50, patience=10, learning_rate=0.0001, 
                     freeze_layers=False, device='cpu', model_name='Model'):
    """
    전이학습 (Fine-tuning)
    """
    print(f"\n{'='*80}")
    print(f"{model_name} 전이학습 시작")
    print(f"{'='*80}")
    print(f"학습률: {learning_rate}")
    print(f"레이어 동결: {freeze_layers}")
    
    # 레이어 동결 옵션
    if freeze_layers:
        # LSTM/GRU 레이어는 동결하고 FC 레이어만 학습
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        print("⚠️  순환 레이어 동결, FC 레이어만 학습")
    else:
        # 모든 레이어 학습
        for param in model.parameters():
            param.requires_grad = True
        print("✅ 전체 레이어 미세조정")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    
    elapsed_time = time.time() - start_time
    print(f"{model_name} 전이학습 완료! 소요 시간: {elapsed_time:.2f}초")
    
    return model, train_losses, val_losses


# === 예측 함수 ===
def predict(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    return np.array(predictions), np.array(actuals)


# === 미래 예측 함수 ===
def predict_future(model, scaler_X, scaler_y, last_sequence, 
                   target_datetime, solar_capacity, device='cpu'):
    """
    특정 시간의 발전량 예측
    """
    model.eval()
    
    # 기상 데이터 생성 (대구 기준)
    month = target_datetime.month
    hour = target_datetime.hour
    
    # 계절별 기상 패턴
    if month in [11, 12, 1, 2]:
        base_temp = 5
        base_humidity = 60
        base_cloud = 5
    elif month in [3, 4, 5]:
        base_temp = 15
        base_humidity = 55
        base_cloud = 4
    elif month in [6, 7, 8]:
        base_temp = 25
        base_humidity = 70
        base_cloud = 6
    else:
        base_temp = 15
        base_humidity = 65
        base_cloud = 5
    
    # 시간대별 온도 조정
    if 6 <= hour <= 12:
        temperature = base_temp + (hour - 6) * 1.5
    elif 12 < hour <= 18:
        temperature = base_temp + 9 - (hour - 12) * 1.0
    else:
        temperature = base_temp - 3
    
    # 일조시간 및 일사량
    if 6 <= hour <= 18:
        sunshine_duration = 0.8 if 9 <= hour <= 15 else 0.3
        solar_radiation = 600 if 9 <= hour <= 15 else 200
    else:
        sunshine_duration = 0
        solar_radiation = 0
    
    # ⭐ 제주 모델과 동일한 8개 특성만 생성
    # feature_cols = ['temperature', 'precipitation', 'humidity', 'cloud_cover',
    #                 'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour']
    new_features = np.array([[
        temperature,           # temperature
        0,                     # precipitation
        base_humidity,         # humidity
        base_cloud,            # cloud_cover
        sunshine_duration,     # sunshine_duration
        solar_radiation,       # solar_radiation
        solar_capacity,        # solar_capacity
        hour                   # hour
    ]])
    
    # 스케일링
    new_features_scaled = scaler_X.transform(new_features)
    
    # 시퀀스 업데이트
    new_sequence = np.vstack([last_sequence[1:], new_features_scaled])
    new_sequence_tensor = torch.FloatTensor(new_sequence).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        prediction_scaled = model(new_sequence_tensor).cpu().numpy()
        prediction = scaler_y.inverse_transform(prediction_scaled)[0, 0]
    
    return max(0, prediction), new_sequence

# === 전이학습 모델 저장 함수 수정 ===
def save_transfer_models(lstm_model, gru_model, scaler_X, scaler_y, 
                        lstm_metrics, gru_metrics, 
                        region_name, model_dir='./saved_models'):
    """
    전이학습된 모델 저장
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(model_dir, f'transfer_{region_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    # ⭐ Metrics를 JSON 직렬화 가능한 형태로 변환
    def convert_to_native(obj):
        """NumPy 타입을 Python 네이티브 타입으로 변환"""
        if isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    lstm_metrics_native = convert_to_native(lstm_metrics)
    gru_metrics_native = convert_to_native(gru_metrics)
    
    # LSTM 모델 저장
    lstm_path = os.path.join(save_dir, f'lstm_transfer_{region_name}_{timestamp}.pth')
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'model_config': {
            'input_size': lstm_model.lstm.input_size,
            'hidden_size': lstm_model.hidden_size,
            'num_layers': lstm_model.num_layers
        },
        'metrics': lstm_metrics_native,
        'timestamp': timestamp,
        'region': region_name
    }, lstm_path)
    print(f"✅ LSTM 전이학습 모델 저장: {lstm_path}")
    
    # GRU 모델 저장
    gru_path = os.path.join(save_dir, f'gru_transfer_{region_name}_{timestamp}.pth')
    torch.save({
        'model_state_dict': gru_model.state_dict(),
        'model_config': {
            'input_size': gru_model.gru.input_size,
            'hidden_size': gru_model.hidden_size,
            'num_layers': gru_model.num_layers
        },
        'metrics': gru_metrics_native,
        'timestamp': timestamp,
        'region': region_name
    }, gru_path)
    print(f"✅ GRU 전이학습 모델 저장: {gru_path}")
    
    # 스케일러 저장
    scaler_path = os.path.join(save_dir, f'scalers_{region_name}_{timestamp}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    print(f"✅ 스케일러 저장: {scaler_path}")
    
    # 메타데이터 저장
    metadata = {
        'timestamp': timestamp,
        'region': region_name,
        'lstm_metrics': lstm_metrics_native,
        'gru_metrics': gru_metrics_native,
        'models': {
            'lstm': lstm_path,
            'gru': gru_path,
            'scalers': scaler_path
        }
    }
    
    metadata_path = os.path.join(save_dir, f'metadata_{region_name}_{timestamp}.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ 메타데이터 저장: {metadata_path}")
    
    # 최신 모델 정보 저장
    latest_path = os.path.join(save_dir, f'latest_model_{region_name}.json')
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': timestamp, 'region': region_name}, f, indent=2)
    
    return timestamp

# === 메인 실행 ===
if __name__ == "__main__":
    try:
        # 1. 제주 사전학습 모델 로드
        lstm_pretrained, gru_pretrained, jeju_metadata = load_jeju_pretrained_models(
            model_dir=model_dir
        )
        
        print(f"\n제주 모델 성능:")
        print(f"  LSTM R²: {jeju_metadata['lstm_metrics']['r2']:.4f}")
        print(f"  GRU R²: {jeju_metadata['gru_metrics']['r2']:.4f}")
        print(f"  Stacking R²: {jeju_metadata['stacked_metrics']['r2']:.4f}")
        
        # 2. 대구 데이터 로딩
        SEQUENCE_LENGTH = 24
        daegu_csv_path = "./dataset/output_by_region/대구.csv"  
        
        (X_train, X_val, X_test, y_train, y_val, y_test,
         scaler_X, scaler_y, feature_cols, df_valid, date_test) = load_daegu_data(
            daegu_csv_path, SEQUENCE_LENGTH
        )
        
        # 3. DataLoader 생성
        BATCH_SIZE = 32
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # 4. LSTM 전이학습
        print("\n" + "="*80)
        print("LSTM 전이학습 (대구)")
        print("="*80)
        
        criterion = nn.MSELoss()
        lstm_model, lstm_train_losses, lstm_val_losses = transfer_learning(
            lstm_pretrained, train_loader, val_loader, criterion,
            num_epochs=50, patience=10, learning_rate=0.0001,
            freeze_layers=False, device=device, model_name='LSTM'
        )
        
        # 5. GRU 전이학습
        print("\n" + "="*80)
        print("GRU 전이학습 (대구)")
        print("="*80)
        
        gru_model, gru_train_losses, gru_val_losses = transfer_learning(
            gru_pretrained, train_loader, val_loader, criterion,
            num_epochs=50, patience=10, learning_rate=0.0001,
            freeze_layers=False, device=device, model_name='GRU'
        )
        
        # 6. 미래 발전량 예측 (24H, 48H, 72H)
        print("\n" + "="*80)
        print("미래 발전량 예측 (대구) - 전이학습 모델 사용")
        print("="*80)
        
        solar_capacity = df_valid['solar_capacity'].iloc[0]
        current_time = datetime.now()  # 실제 현재 시각 사용
        last_sequence = X_test[-1]  # 마지막 시퀀스 사용
        
        print(f"\n📍 현재 시각: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 설비용량: {solar_capacity:.2f} MW")
        print(f"\n전이학습된 LSTM + GRU 모델을 사용하여 미래 발전량을 예측합니다.")
        
        # 24H, 48H, 72H 후 예측
        for hours_ahead in [24, 48, 72]:
            target_date = current_time + timedelta(hours=hours_ahead)
            print(f"\n{'='*70}")
            print(f"📅 {hours_ahead}시간 후 예측: {target_date.strftime('%Y-%m-%d %A')}")
            print(f"{'='*70}")
            
            daily_predictions_lstm = []
            daily_predictions_gru = []
            daily_predictions_ensemble = []
            temp_sequence = last_sequence.copy()
            hourly_details = []
            
            for h in range(24):
                target_time = current_time + timedelta(hours=hours_ahead+h)
                
                # LSTM 예측
                lstm_pred, temp_sequence = predict_future(
                    lstm_model, scaler_X, scaler_y, temp_sequence,
                    target_time, solar_capacity, device
                )
                
                # GRU 예측
                gru_pred, _ = predict_future(
                    gru_model, scaler_X, scaler_y, temp_sequence,
                    target_time, solar_capacity, device
                )
                
                # 앙상블 (LSTM + GRU 평균)
                ensemble_pred = (lstm_pred + gru_pred) / 2
                
                daily_predictions_lstm.append(max(0, lstm_pred))
                daily_predictions_gru.append(max(0, gru_pred))
                daily_predictions_ensemble.append(max(0, ensemble_pred))
                
                # 시간별 상세 정보 저장
                hourly_details.append({
                    'time': target_time.strftime('%H:%M'),
                    'lstm': lstm_pred,
                    'gru': gru_pred,
                    'ensemble': ensemble_pred
                })
            
            # 일일 통계 계산
            total_lstm = sum(daily_predictions_lstm)
            total_gru = sum(daily_predictions_gru)
            total_ensemble = sum(daily_predictions_ensemble)
            
            peak_lstm = max(daily_predictions_lstm)
            peak_gru = max(daily_predictions_gru)
            peak_ensemble = max(daily_predictions_ensemble)
            
            # 결과 출력
            print(f"\n[LSTM 모델 예측]")
            print(f"  일일 총 발전량: {total_lstm:.2f} MWh")
            print(f"  피크 발전량: {peak_lstm:.2f} MWh (시간당)")
            print(f"  평균 시간당: {total_lstm/24:.2f} MWh")
            print(f"  평균 가동률: {(total_lstm/(solar_capacity*24))*100:.1f}%")
            
            print(f"\n[GRU 모델 예측]")
            print(f"  일일 총 발전량: {total_gru:.2f} MWh")
            print(f"  피크 발전량: {peak_gru:.2f} MWh (시간당)")
            print(f"  평균 시간당: {total_gru/24:.2f} MWh")
            print(f"  평균 가동률: {(total_gru/(solar_capacity*24))*100:.1f}%")
            
            print(f"\n[앙상블 예측 (LSTM+GRU 평균)] ⭐ 권장")
            print(f"  일일 총 발전량: {total_ensemble:.2f} MWh")
            print(f"  피크 발전량: {peak_ensemble:.2f} MWh (시간당)")
            print(f"  평균 시간당: {total_ensemble/24:.2f} MWh")
            print(f"  평균 가동률: {(total_ensemble/(solar_capacity*24))*100:.1f}%")
            
            # 시간별 상세 예측 (주요 발전 시간대만 출력)
            print(f"\n⏰ 시간별 발전량 상세 (앙상블 기준, 발전량 > 0.5 MWh):")
            print("-" * 60)
            for detail in hourly_details:
                if detail['ensemble'] > 0.5:
                    print(f"  {detail['time']} - {detail['ensemble']:6.2f} MWh "
                          f"(LSTM: {detail['lstm']:5.2f}, GRU: {detail['gru']:5.2f})")
        
        # 7. 최종 요약
        
        # 7. 최종 요약
        print(f"\n{'='*80}")
        print("대구 전이학습 및 미래 예측 완료!")
        print(f"{'='*80}")

        # 기존 코드의 6번 섹션 이후에 추가

        # 7. 테스트 세트 평가 및 모델 저장
        print("\n" + "="*80)
        print("테스트 세트 평가")
        print("="*80)
        
        # LSTM 평가
        lstm_predictions, lstm_actuals = predict(lstm_model, test_loader, device)
        lstm_predictions_original = scaler_y.inverse_transform(lstm_predictions)
        lstm_actuals_original = scaler_y.inverse_transform(lstm_actuals)
        
        lstm_metrics = calculate_all_metrics(
            lstm_actuals_original, lstm_predictions_original, print_details=True
        )
        
        print("\n[LSTM 전이학습 모델 성능]")
        print(f"  MAE: {lstm_metrics['mae']:.4f} MWh")
        print(f"  RMSE: {lstm_metrics['rmse']:.4f} MWh")
        print(f"  R²: {lstm_metrics['r2']:.4f}")
        print(f"  MAPE: {lstm_metrics['mape']:.2f}%")
        
        # GRU 평가
        gru_predictions, gru_actuals = predict(gru_model, test_loader, device)
        gru_predictions_original = scaler_y.inverse_transform(gru_predictions)
        gru_actuals_original = scaler_y.inverse_transform(gru_actuals)
        
        gru_metrics = calculate_all_metrics(
            gru_actuals_original, gru_predictions_original, print_details=True
        )
        
        print("\n[GRU 전이학습 모델 성능]")
        print(f"  MAE: {gru_metrics['mae']:.4f} MWh")
        print(f"  RMSE: {gru_metrics['rmse']:.4f} MWh")
        print(f"  R²: {gru_metrics['r2']:.4f}")
        print(f"  MAPE: {gru_metrics['mape']:.2f}%")
        
        # 앙상블 평가
        ensemble_predictions = (lstm_predictions_original + gru_predictions_original) / 2
        ensemble_metrics = calculate_all_metrics(
            lstm_actuals_original, ensemble_predictions, print_details=True
        )
        
        print("\n[앙상블 (LSTM+GRU) 성능]")
        print(f"  MAE: {ensemble_metrics['mae']:.4f} MWh")
        print(f"  RMSE: {ensemble_metrics['rmse']:.4f} MWh")
        print(f"  R²: {ensemble_metrics['r2']:.4f}")
        print(f"  MAPE: {ensemble_metrics['mape']:.2f}%")
        
        # 8. 전이학습 모델 저장
        print("\n" + "="*80)
        print("전이학습 모델 저장")
        print("="*80)
        
        saved_timestamp = save_transfer_models(
            lstm_model, gru_model, scaler_X, scaler_y,
            lstm_metrics, gru_metrics,
            region_name='daegu',
            model_dir=model_dir
        )
        
        print(f"\n✅ 대구 전이학습 모델 저장 완료! (Timestamp: {saved_timestamp})")
        
    except FileNotFoundError as e:
        print(f"Error: 파일을 찾을 수 없습니다. {e}")
        print("\n확인 사항:")
        print("1. 제주 사전학습 모델이 ./saved_models 디렉토리에 있는지 확인")
        print("2. 대구 CSV 파일 경로가 올바른지 확인")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()