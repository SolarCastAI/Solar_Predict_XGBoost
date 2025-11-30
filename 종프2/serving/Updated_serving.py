import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings
import os
import json
import pickle
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

# --- 기본 설정 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
model_dir = "./saved_models/transfer_daegu"

# ==========================================
# 1. Dataset 클래스
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. AI 모델 클래스 정의 (LSTM, GRU)
# ==========================================
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


# ==========================================
# 3. 평가 지표 함수들
# ==========================================
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


# ==========================================
# 4. 모델 로드 함수
# ==========================================
def load_daegu_transfer_models(model_dir='./saved_models/transfer_daegu', timestamp=None):
    """대구 전이학습 모델 로드"""
    
    # 최신 모델 정보 로드
    if timestamp is None:
        latest_path = os.path.join(model_dir, 'latest_model_daegu.json')
        if not os.path.exists(latest_path):
            # latest 파일이 없으면 디렉토리에서 가장 최근 메타데이터 찾기
            metadata_files = [f for f in os.listdir(model_dir) 
                            if f.startswith('metadata_daegu_') and f.endswith('.json')]
            if not metadata_files:
                raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_dir}")
            
            metadata_files.sort(reverse=True)
            metadata_path = os.path.join(model_dir, metadata_files[0])
            timestamp = metadata_files[0].replace('metadata_daegu_', '').replace('.json', '')
        else:
            with open(latest_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            timestamp = model_info['timestamp']
            metadata_path = os.path.join(model_dir, f'metadata_daegu_{timestamp}.json')
    else:
        metadata_path = os.path.join(model_dir, f'metadata_daegu_{timestamp}.json')
    
    # 메타데이터 로드
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # LSTM 모델 로드
    lstm_path = os.path.join(model_dir, f'lstm_transfer_daegu_{timestamp}.pth')
    lstm_checkpoint = torch.load(lstm_path, map_location=device, weights_only=False)
    lstm_config = lstm_checkpoint['model_config']
    
    lstm_model = LSTMModel(
        input_size=lstm_config['input_size'],
        hidden_size=lstm_config['hidden_size'],
        num_layers=lstm_config['num_layers']
    ).to(device)
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    
    # GRU 모델 로드
    gru_path = os.path.join(model_dir, f'gru_transfer_daegu_{timestamp}.pth')
    gru_checkpoint = torch.load(gru_path, map_location=device, weights_only=False)
    gru_config = gru_checkpoint['model_config']
    
    gru_model = GRUModel(
        input_size=gru_config['input_size'],
        hidden_size=gru_config['hidden_size'],
        num_layers=gru_config['num_layers']
    ).to(device)
    gru_model.load_state_dict(gru_checkpoint['model_state_dict'])
    
    # 스케일러 로드
    scaler_path = os.path.join(model_dir, f'scalers_daegu_{timestamp}.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return lstm_model, gru_model, scalers['scaler_X'], scalers['scaler_y'], metadata


# ==========================================
# 5. 데이터 전처리 함수
# ==========================================
def preprocess_data_from_db(df, sequence_length=24):
    """
    [추론용] DB 데이터를 모델 입력 형태로 변환
    부산 데이터 형식에 맞춰 전처리
    """
    df_renamed = df.copy()
    
    # 날짜 처리
    if 'datetime' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['datetime'])
    elif '발전일자' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['발전일자'])
        
        # 컬럼 매핑 (부산 데이터 형식)
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
        df_renamed = df_renamed.rename(columns=column_mapping)
    
    # 결측치 처리
    df_renamed['precipitation'] = df_renamed['precipitation'].fillna(0)
    df_renamed['snow'] = df_renamed['snow'].fillna(0)
    df_renamed['sunshine_duration'] = df_renamed['sunshine_duration'].fillna(0)
    df_renamed['solar_radiation'] = df_renamed['solar_radiation'].fillna(0)
    df_renamed['humidity'] = df_renamed['humidity'].fillna(df_renamed['humidity'].mean())
    df_renamed['temperature'] = df_renamed['temperature'].fillna(df_renamed['temperature'].mean())
    
    # cloud_cover 처리
    if 'cloud_cover' not in df_renamed.columns:
        if '적운량(3분위)' in df.columns:
            df_renamed['cloud_cover'] = df['적운량(3분위)'].fillna(1) * 3.33
        else:
            df_renamed['cloud_cover'] = 5
    else:
        df_renamed['cloud_cover'] = df_renamed['cloud_cover'].fillna(5)
    
    # 시간 특성 추가
    df_renamed['hour'] = df_renamed['datetime'].dt.hour
    
    # 대구 모델과 동일한 8개 특성 사용
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'cloud_cover',
        'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour'
    ]
    
    # 유효한 데이터만 선택
    target_col = 'solar_generation'
    df_valid = df_renamed[df_renamed[target_col].notna()].copy()
    
    if len(df_valid) < sequence_length + 1:
        return None, None, None, feature_cols, None
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values.reshape(-1, 1)
    
    # 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 시퀀스 생성
    X_seq = []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
    
    X_seq = np.array(X_seq)
    
    return X_seq, scaler_X, scaler_y, feature_cols, df_valid


def preprocess_train_data_from_db(df, sequence_length=24):
    """
    [재학습용] 데이터 전처리 및 Train/Val/Test 분할
    """
    df_renamed = df.copy()
    
    # 날짜 처리
    if 'datetime' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['datetime'])
    elif '발전일자' in df_renamed.columns:
        df_renamed['datetime'] = pd.to_datetime(df_renamed['발전일자'])
        
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
        df_renamed = df_renamed.rename(columns=column_mapping)
    
    # 결측치 처리
    df_renamed['precipitation'] = df_renamed['precipitation'].fillna(0)
    df_renamed['snow'] = df_renamed['snow'].fillna(0)
    df_renamed['sunshine_duration'] = df_renamed['sunshine_duration'].fillna(0)
    df_renamed['solar_radiation'] = df_renamed['solar_radiation'].fillna(0)
    df_renamed['humidity'] = df_renamed['humidity'].fillna(df_renamed['humidity'].mean())
    df_renamed['temperature'] = df_renamed['temperature'].fillna(df_renamed['temperature'].mean())
    
    if 'cloud_cover' not in df_renamed.columns:
        if '적운량(3분위)' in df.columns:
            df_renamed['cloud_cover'] = df['적운량(3분위)'].fillna(1) * 3.33
        else:
            df_renamed['cloud_cover'] = 5
    else:
        df_renamed['cloud_cover'] = df_renamed['cloud_cover'].fillna(5)
    
    df_renamed['hour'] = df_renamed['datetime'].dt.hour
    
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'cloud_cover',
        'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour'
    ]
    
    target_col = 'solar_generation'
    df_valid = df_renamed[df_renamed[target_col].notna()].copy()
    
    if len(df_valid) < sequence_length + 10:
        return None, None, None, None, None, None, None, None, feature_cols, None, None
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values.reshape(-1, 1)
    dates = df_valid['datetime'].values
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 시퀀스 생성
    X_seq, y_seq, date_seq = [], [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
        date_seq.append(dates[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Train/Val/Test 분할 (80/10/10)
    X_temp, X_test, y_temp, y_test, date_temp, date_test = train_test_split(
        X_seq, y_seq, date_seq, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val, date_train, date_val = train_test_split(
        X_temp, y_temp, date_temp, test_size=0.111, random_state=42
    )
    
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            scaler_X, scaler_y, feature_cols, df_valid, date_test)


# ==========================================
# 6. 미래 예측 함수
# ==========================================
def predict_future_single_step(model, scaler_X, scaler_y, last_sequence, 
                                target_time, solar_capacity, device='cpu'):
    """단일 시점 미래 예측 (1시간 후)"""
    model.eval()
    
    # 마지막 시퀀스의 평균값으로 다음 시점 특성 추정
    last_features = last_sequence[-1].copy()
    
    # 시간 특성 업데이트
    target_hour = target_time.hour
    hour_scaled = target_hour / 23.0
    last_features[7] = hour_scaled  # hour는 8번째 특성
    
    # 새 시퀀스 생성 (sliding window)
    new_sequence = np.vstack([last_sequence[1:], last_features.reshape(1, -1)])
    
    # 예측
    with torch.no_grad():
        X_tensor = torch.FloatTensor(new_sequence).unsqueeze(0).to(device)
        pred_scaled = model(X_tensor).cpu().numpy()
        pred_original = scaler_y.inverse_transform(pred_scaled)[0, 0]
    
    return max(0, pred_original), new_sequence


# ==========================================
# 7. 전이학습 함수
# ==========================================
def transfer_learning(model, train_loader, val_loader, criterion, 
                     num_epochs=10, patience=3, learning_rate=0.0001, 
                     freeze_layers=False, device='cpu', model_name='Model'):
    """전이학습 (Fine-tuning) 수행"""
    
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    model.load_state_dict(best_model_state)
    return model


# ==========================================
# 8. Main Entry Points (Celery Task에서 호출)
# ==========================================
def run_prediction(df_input, loaded_models=None):
    """
    [1시간 주기] 예측 실행 함수
    
    Args:
        df_input: DB에서 가져온 DataFrame
        loaded_models: (lstm_model, gru_model, scaler_X, scaler_y) 튜플 (선택사항)
    
    Returns:
        list: 72시간 예측 결과 (시간별 현재/누적 발전량 포함)
    """
    try:
        # 1. 모델 로드
        if loaded_models:
            lstm_model, gru_model, scaler_X, scaler_y = loaded_models
        else:
            lstm_model, gru_model, scaler_X, scaler_y, _ = load_daegu_transfer_models(
                model_dir=model_dir
            )
        
        SEQUENCE_LENGTH = 24
        
        # 2. 데이터 전처리
        if df_input.empty:
            return []
        
        X_seq, _, _, _, df_valid = preprocess_data_from_db(df_input, SEQUENCE_LENGTH)
        
        if X_seq is None or len(X_seq) == 0:
            print("⚠️ 시퀀스를 만들 데이터가 부족합니다.")
            return []
        
        # 3. 현재 시간 기준 설정
        current_time = datetime.now()
        solar_capacity = df_valid['solar_capacity'].iloc[0]
        last_sequence = X_seq[-1]
        
        # 4. 72시간 예측 수행
        all_predictions = []
        lstm_cumulative = 0
        gru_cumulative = 0
        ensemble_cumulative = 0
        
        temp_sequence = last_sequence.copy()
        
        for h in range(1, 73):
            target_time = current_time + timedelta(hours=h)
            
            # LSTM 예측
            lstm_pred, temp_sequence = predict_future_single_step(
                lstm_model, scaler_X, scaler_y, temp_sequence,
                target_time, solar_capacity, device
            )
            
            # GRU 예측
            gru_pred, _ = predict_future_single_step(
                gru_model, scaler_X, scaler_y, temp_sequence,
                target_time, solar_capacity, device
            )
            
            # 앙상블
            ensemble_pred = (lstm_pred + gru_pred) / 2
            
            # 누적 발전량 업데이트
            lstm_cumulative += lstm_pred
            gru_cumulative += gru_pred
            ensemble_cumulative += ensemble_pred
            
            all_predictions.append({
                '예측_날짜': target_time.strftime('%Y-%m-%d'),
                '예측_시간': target_time.strftime('%H:%M'),
                '예측_일시': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                '경과_시간(H)': h,
                'LSTM_현재_발전량(MWh)': round(lstm_pred, 4),
                'LSTM_누적_발전량(MWh)': round(lstm_cumulative, 4),
                'GRU_현재_발전량(MWh)': round(gru_pred, 4),
                'GRU_누적_발전량(MWh)': round(gru_cumulative, 4),
                '앙상블_현재_발전량(MWh)': round(ensemble_pred, 4),
                '앙상블_누적_발전량(MWh)': round(ensemble_cumulative, 4)
            })
        
        return all_predictions
    
    except Exception as e:
        print(f"❌ 예측 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return []


def retrain_model(df_train):
    """
    [하루 1번] 재학습 실행 함수
    
    Args:
        df_train: 학습용 DataFrame
    
    Returns:
        bool: 재학습 성공 여부
    """
    try:
        print("\n🚀 [Model Retraining] 시작...")
        
        # 1. 기존 모델 로드
        lstm_model, gru_model, _, _, metadata = load_daegu_transfer_models(model_dir=model_dir)
        
        SEQUENCE_LENGTH = 24
        
        # 2. 데이터 전처리
        result = preprocess_train_data_from_db(df_train, SEQUENCE_LENGTH)
        
        if result[0] is None:
            print("⚠️ 학습할 데이터가 너무 적습니다.")
            return False
        
        (X_train, X_val, X_test, y_train, y_val, y_test,
         scaler_X, scaler_y, feature_cols, df_valid, date_test) = result
        
        # 3. DataLoader 생성
        BATCH_SIZE = 32
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # 4. 전이학습 수행
        criterion = nn.MSELoss()
        
        print("   >> LSTM 전이학습 중...")
        lstm_model = transfer_learning(
            lstm_model, train_loader, val_loader, criterion,
            num_epochs=10, patience=3, learning_rate=0.0001,
            freeze_layers=False, device=device, model_name='LSTM'
        )
        
        print("   >> GRU 전이학습 중...")
        gru_model = transfer_learning(
            gru_model, train_loader, val_loader, criterion,
            num_epochs=10, patience=3, learning_rate=0.0001,
            freeze_layers=False, device=device, model_name='GRU'
        )
        
        # 5. 모델 및 스케일러 저장
        new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 모델 config 가져오기
        lstm_config = {
            'input_size': 8,
            'hidden_size': lstm_model.hidden_size,
            'num_layers': lstm_model.num_layers,
            'dropout': 0.2
        }
        
        gru_config = {
            'input_size': 8,
            'hidden_size': gru_model.hidden_size,
            'num_layers': gru_model.num_layers,
            'dropout': 0.2
        }
        
        # LSTM 저장
        torch.save({
            'model_state_dict': lstm_model.state_dict(),
            'model_config': lstm_config
        }, os.path.join(model_dir, f'lstm_transfer_daegu_{new_timestamp}.pth'))
        
        # GRU 저장
        torch.save({
            'model_state_dict': gru_model.state_dict(),
            'model_config': gru_config
        }, os.path.join(model_dir, f'gru_transfer_daegu_{new_timestamp}.pth'))
        
        # 스케일러 저장
        with open(os.path.join(model_dir, f'scalers_daegu_{new_timestamp}.pkl'), 'wb') as f:
            pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
        
        # 메타데이터 갱신
        metadata['timestamp'] = new_timestamp
        metadata['retrained_at'] = datetime.now().isoformat()
        metadata['lstm_config'] = lstm_config
        metadata['gru_config'] = gru_config
        
        with open(os.path.join(model_dir, f'metadata_daegu_{new_timestamp}.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        # latest 파일 갱신
        with open(os.path.join(model_dir, 'latest_model_daegu.json'), 'w', encoding='utf-8') as f:
            json.dump({'timestamp': new_timestamp}, f, indent=4)
        
        print(f"✅ 재학습 완료! 새로운 모델 버전: {new_timestamp}")
        return True
    
    except Exception as e:
        print(f"❌ 재학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# 9. 모델 사전 로드 함수 (선택사항)
# ==========================================
def preload_models():
    """
    서버 시작 시 모델을 미리 로드하여 메모리에 유지
    매 요청마다 로드하는 오버헤드 제거
    
    Returns:
        tuple: (lstm_model, gru_model, scaler_X, scaler_y)
    """
    try:
        print("🔥 모델 사전 로드 중...")
        lstm_model, gru_model, scaler_X, scaler_y, metadata = load_daegu_transfer_models(
            model_dir=model_dir
        )
        print(f"✅ 모델 로드 완료 (버전: {metadata.get('timestamp', 'unknown')})")
        return lstm_model, gru_model, scaler_X, scaler_y
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

'''

# ==========================================
# 사용 예시
# ==========================================
if __name__ == "__main__":
    # 테스트용 코드
    print(f"🚀 사용 중인 디바이스: {device}")
    
    # 모델 사전 로드 테스트
    models = preload_models()
    
    if models:
        print("\n✅ 모델이 정상적으로 로드되었습니다.")
        print("   이제 run_prediction() 또는 retrain_model()을 호출할 수 있습니다.")
        
'''