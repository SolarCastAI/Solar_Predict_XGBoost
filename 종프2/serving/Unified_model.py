import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# GPU/CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 사용 중인 디바이스: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA를 사용할 수 없습니다. CPU를 사용합니다.")


# === LSTM 모델 정의 ===
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


# === GRU 모델 정의 ===
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


# === PyTorch Dataset ===
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

def calculate_all_metrics(y_true, y_pred, print_details=True):
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
        print(f"\n{'='*60}")
        print(f"📊 평가 지표")
        print(f"{'='*60}")
        print(f"MAE (평균 절대 오차):     {metrics['mae']:.4f} MWh")
        print(f"RMSE (평균 제곱근 오차):  {metrics['rmse']:.4f} MWh")
        print(f"NMAE (정규화 MAE):        {metrics['nmae']:.4f}")
        print(f"NRMSE (정규화 RMSE):      {metrics['nrmse']:.4f}")
        print(f"R² (결정계수):            {metrics['r2']:.4f}")
        print(f"MAPE (평균 절대 백분율):  {metrics['mape']:.2f}%")
        print(f"{'='*60}")
    
    return metrics


# === 부산 데이터 로딩 함수 ===
def load_busan_data(file_path, sequence_length=24):
    """
    부산 CSV 데이터 로딩 및 전처리
    """
    print("\n" + "="*80)
    print("부산 데이터 로딩 중...")
    print("="*80)
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"원본 데이터 크기: {df.shape}")
    print(f"컬럼: {df.columns.tolist()}")
    
    # 날짜 처리
    df['발전일자'] = pd.to_datetime(df['발전일자'])
    
    # 컬럼 매핑 (부산 데이터 형식에 맞게)
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
    
    # cloud_cover 처리 (적운량(10분위) 사용)
    if 'cloud_cover' in df_renamed.columns:
        df_renamed['cloud_cover'] = df_renamed['cloud_cover'].fillna(5)
    else:
        # 적운량(3분위)가 있는 경우 10분위로 변환
        if '적운량(3분위)' in df.columns:
            df_renamed['cloud_cover'] = df['적운량(3분위)'].fillna(1) * 3.33
        else:
            df_renamed['cloud_cover'] = 5  # 기본값
    
    # 시간 특성 추가
    df_renamed['hour'] = df_renamed['datetime'].dt.hour
    df_renamed['month'] = df_renamed['datetime'].dt.month
    df_renamed['day_of_year'] = df_renamed['datetime'].dt.dayofyear
    
    print(f"\n데이터 기간: {df_renamed['datetime'].min()} ~ {df_renamed['datetime'].max()}")
    print(f"평균 발전량: {df_renamed['solar_generation'].mean():.2f} MWh")
    print(f"설비용량: {df_renamed['solar_capacity'].iloc[0]:.2f} MW")
    
    # ⭐ 제주/대구 모델과 동일한 8개 특성만 선택
    feature_cols = [
        'temperature', 'precipitation', 'humidity', 'cloud_cover',
        'sunshine_duration', 'solar_radiation', 'solar_capacity', 'hour'
    ]
    
    print(f"\n사용 특성 ({len(feature_cols)}개):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
    
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


# === 대구 전이학습 모델 로드 함수 ===
def load_daegu_transfer_models(model_dir='./saved_models/transfer_daegu', timestamp=None):
    """
    대구 전이학습 모델 로드
    """
    print(f"\n{'='*80}")
    print("대구 전이학습 모델 로드 중...")
    print(f"{'='*80}")
    
    # 최신 모델 정보 로드
    if timestamp is None:
        latest_path = os.path.join(model_dir, 'latest_model_daegu.json')
        if not os.path.exists(latest_path):
            # latest 파일이 없으면 디렉토리에서 가장 최근 메타데이터 찾기
            metadata_files = [f for f in os.listdir(model_dir) if f.startswith('metadata_daegu_') and f.endswith('.json')]
            if not metadata_files:
                raise FileNotFoundError(f"모델을 찾을 수 없습니다: {model_dir}")
            
            # 가장 최근 파일 선택
            metadata_files.sort(reverse=True)
            metadata_path = os.path.join(model_dir, metadata_files[0])
            timestamp = metadata_files[0].replace('metadata_daegu_', '').replace('.json', '')
            print(f"⚠️  latest 파일 없음. 가장 최근 모델 사용: {timestamp}")
        else:
            with open(latest_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            timestamp = model_info['timestamp']
            print(f"최신 모델 타임스탬프: {timestamp}")
            metadata_path = os.path.join(model_dir, f'metadata_daegu_{timestamp}.json')
    else:
        metadata_path = os.path.join(model_dir, f'metadata_daegu_{timestamp}.json')
    
    # 메타데이터 로드
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"✅ 메타데이터 로드: {metadata_path}")
    
    # LSTM 모델 로드
    lstm_path = os.path.join(model_dir, f'lstm_transfer_daegu_{timestamp}.pth')
    lstm_checkpoint = torch.load(lstm_path, map_location=device)
    lstm_config = lstm_checkpoint['model_config']
    
    lstm_model = LSTMModel(
        input_size=lstm_config['input_size'],
        hidden_size=lstm_config['hidden_size'],
        num_layers=lstm_config['num_layers']
    ).to(device)
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    print(f"✅ LSTM 전이학습 모델 로드: {lstm_path}")
    
    # GRU 모델 로드
    gru_path = os.path.join(model_dir, f'gru_transfer_daegu_{timestamp}.pth')
    gru_checkpoint = torch.load(gru_path, map_location=device)
    gru_config = gru_checkpoint['model_config']
    
    gru_model = GRUModel(
        input_size=gru_config['input_size'],
        hidden_size=gru_config['hidden_size'],
        num_layers=gru_config['num_layers']
    ).to(device)
    gru_model.load_state_dict(gru_checkpoint['model_state_dict'])
    print(f"✅ GRU 전이학습 모델 로드: {gru_path}")
    
    # 스케일러 로드
    scaler_path = os.path.join(model_dir, f'scalers_daegu_{timestamp}.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print(f"✅ 스케일러 로드: {scaler_path}")
    
    print(f"\n대구 모델 학습 성능 (메타데이터):")
    print(f"  LSTM R²: {metadata['lstm_metrics']['r2']:.4f}")
    print(f"  LSTM MAPE: {metadata['lstm_metrics']['mape']:.2f}%")
    print(f"  GRU R²: {metadata['gru_metrics']['r2']:.4f}")
    print(f"  GRU MAPE: {metadata['gru_metrics']['mape']:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"✨ 대구 전이학습 모델 로드 완료!")
    print(f"{'='*80}")
    
    return lstm_model, gru_model, scalers['scaler_X'], scalers['scaler_y'], metadata


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


# ==========================================
# ⭐ 미래 예측 함수 (현재 날짜 기준, 시간별 누적 발전량 포함)
# ==========================================
def predict_future_single_step(model, scaler_X, scaler_y, last_sequence, 
                                target_time, solar_capacity, device='cpu'):
    """
    단일 시점 미래 예측 (1시간 후)
    """
    model.eval()
    
    # 마지막 시퀀스의 평균값으로 다음 시점 특성 추정
    last_features = last_sequence[-1].copy()
    
    # 시간 특성 업데이트 (hour)
    target_hour = target_time.hour
    hour_scaled = target_hour / 23.0  # MinMax 스케일링 근사
    last_features[7] = hour_scaled  # hour는 8번째 특성
    
    # 새 시퀀스 생성 (sliding window)
    new_sequence = np.vstack([last_sequence[1:], last_features.reshape(1, -1)])
    
    # 예측
    with torch.no_grad():
        X_tensor = torch.FloatTensor(new_sequence).unsqueeze(0).to(device)
        pred_scaled = model(X_tensor).cpu().numpy()
        pred_original = scaler_y.inverse_transform(pred_scaled)[0, 0]
    
    return max(0, pred_original), new_sequence


def generate_future_predictions(lstm_model, gru_model, scaler_X, scaler_y, 
                                 X_test, df_valid, device='cpu'):
    """
    24H, 48H, 72H 미래 예측 생성 (현재 날짜 기준, 시간별 현재/누적 발전량 포함)
    """
    print("\n" + "="*80)
    print("🔮 미래 예측 생성 중 (24H, 48H, 72H)...")
    print("="*80)
    
    # 마지막 시퀀스 선택
    last_sequence = X_test[-1].copy()
    
    # ⭐ 현재 날짜/시간 기준으로 설정
    current_time = datetime.now()
    print(f"현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    solar_capacity = df_valid['solar_capacity'].iloc[0]
    
    predictions_list = []
    
    # 누적 발전량 추적 변수
    lstm_cumulative = 0
    gru_cumulative = 0
    ensemble_cumulative = 0
    
    # 72시간 예측
    for hour in range(1, 73):
        target_time = current_time + timedelta(hours=hour)
        
        # LSTM 예측
        lstm_pred, last_sequence = predict_future_single_step(
            lstm_model, scaler_X, scaler_y, last_sequence,
            target_time, solar_capacity, device
        )
        
        # GRU 예측
        gru_pred, _ = predict_future_single_step(
            gru_model, scaler_X, scaler_y, last_sequence,
            target_time, solar_capacity, device
        )
        
        # 앙상블 (평균)
        ensemble_pred = (lstm_pred + gru_pred) / 2
        
        # 누적 발전량 업데이트
        lstm_cumulative += lstm_pred
        gru_cumulative += gru_pred
        ensemble_cumulative += ensemble_pred
        
        predictions_list.append({
            '예측_날짜': target_time.strftime('%Y-%m-%d'),
            '예측_시간': target_time.strftime('%H:%M'),
            '예측_일시': target_time.strftime('%Y-%m-%d %H:%M:%S'),
            '경과_시간(H)': hour,
            'LSTM_현재_발전량(MWh)': round(lstm_pred, 4),
            'LSTM_누적_발전량(MWh)': round(lstm_cumulative, 4),
            'GRU_현재_발전량(MWh)': round(gru_pred, 4),
            'GRU_누적_발전량(MWh)': round(gru_cumulative, 4),
            '앙상블_현재_발전량(MWh)': round(ensemble_pred, 4),
            '앙상블_누적_발전량(MWh)': round(ensemble_cumulative, 4)
        })
        
        if hour % 24 == 0:
            print(f"  {hour}H 예측 완료 (누적: 앙상블 {ensemble_cumulative:.2f} MWh)")
    
    return pd.DataFrame(predictions_list)


def save_prediction_csvs(predictions_df, output_dir='./prediction_results'):
    """
    24H, 48H, 72H 예측 결과를 별도 CSV 파일로 저장 (시간별 현재/누적 발전량 포함)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 24H 예측
    pred_24h = predictions_df[predictions_df['경과_시간(H)'] <= 24].copy()
    file_24h = os.path.join(output_dir, f'prediction_24H_{timestamp}.csv')
    pred_24h.to_csv(file_24h, index=False, encoding='utf-8-sig')
    print(f"✅ 24H 예측 저장: {file_24h}")
    print(f"   - 총 {len(pred_24h)}개 시간별 데이터")
    print(f"   - 누적 발전량(앙상블): {pred_24h['앙상블_누적_발전량(MWh)'].iloc[-1]:.2f} MWh")
    
    # 48H 예측
    pred_48h = predictions_df[predictions_df['경과_시간(H)'] <= 48].copy()
    file_48h = os.path.join(output_dir, f'prediction_48H_{timestamp}.csv')
    pred_48h.to_csv(file_48h, index=False, encoding='utf-8-sig')
    print(f"✅ 48H 예측 저장: {file_48h}")
    print(f"   - 총 {len(pred_48h)}개 시간별 데이터")
    print(f"   - 누적 발전량(앙상블): {pred_48h['앙상블_누적_발전량(MWh)'].iloc[-1]:.2f} MWh")
    
    # 72H 예측
    pred_72h = predictions_df.copy()
    file_72h = os.path.join(output_dir, f'prediction_72H_{timestamp}.csv')
    pred_72h.to_csv(file_72h, index=False, encoding='utf-8-sig')
    print(f"✅ 72H 예측 저장: {file_72h}")
    print(f"   - 총 {len(pred_72h)}개 시간별 데이터")
    print(f"   - 누적 발전량(앙상블): {pred_72h['앙상블_누적_발전량(MWh)'].iloc[-1]:.2f} MWh")
    
    # 전체 예측 (통합본)
    file_all = os.path.join(output_dir, f'prediction_ALL_{timestamp}.csv')
    predictions_df.to_csv(file_all, index=False, encoding='utf-8-sig')
    print(f"✅ 전체 예측 저장: {file_all}")
    
    return {
        '24H': file_24h,
        '48H': file_48h,
        '72H': file_72h,
        'ALL': file_all
    }


# === 메인 실행 ===
if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("🔥 대구 전이학습 모델의 부산 데이터 성능 평가 + 미래 예측")
        print("="*80)
        
        # 1. 대구 전이학습 모델 로드
        lstm_model, gru_model, daegu_scaler_X, daegu_scaler_y, metadata = load_daegu_transfer_models(
            model_dir='./saved_models/transfer_daegu'
        )
        
        # 2. 부산 데이터 로딩
        SEQUENCE_LENGTH = 24
        busan_csv_path = "./dataset/output_by_region/부산.csv"
        
        if not os.path.exists(busan_csv_path):
            print(f"❌ 파일을 찾을 수 없습니다: {busan_csv_path}")
            print("현재 디렉토리의 CSV 파일을 지정해주세요.")
            exit(1)
        
        (X_train, X_val, X_test, y_train, y_val, y_test,
         scaler_X, scaler_y, feature_cols, df_valid, date_test) = load_busan_data(
            busan_csv_path, SEQUENCE_LENGTH
        )
        
        # 3. DataLoader 생성
        BATCH_SIZE = 32
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # 4. LSTM 모델 평가
        print("\n" + "="*80)
        print("📈 LSTM 모델 평가 (부산 데이터)")
        print("="*80)
        
        lstm_predictions, lstm_actuals = predict(lstm_model, test_loader, device)
        lstm_predictions_original = scaler_y.inverse_transform(lstm_predictions)
        lstm_actuals_original = scaler_y.inverse_transform(lstm_actuals)
        
        print("\n[LSTM 모델 성능 - 부산 데이터]")
        lstm_metrics = calculate_all_metrics(
            lstm_actuals_original, lstm_predictions_original, print_details=True
        )
        
        # 5. GRU 모델 평가
        print("\n" + "="*80)
        print("📈 GRU 모델 평가 (부산 데이터)")
        print("="*80)
        
        gru_predictions, gru_actuals = predict(gru_model, test_loader, device)
        gru_predictions_original = scaler_y.inverse_transform(gru_predictions)
        gru_actuals_original = scaler_y.inverse_transform(gru_actuals)
        
        print("\n[GRU 모델 성능 - 부산 데이터]")
        gru_metrics = calculate_all_metrics(
            gru_actuals_original, gru_predictions_original, print_details=True
        )
        
        # 6. 앙상블 모델 평가
        print("\n" + "="*80)
        print("📈 앙상블 모델 평가 (LSTM + GRU 평균, 부산 데이터)")
        print("="*80)
        
        ensemble_predictions = (lstm_predictions_original + gru_predictions_original) / 2
        
        print("\n[앙상블 모델 성능 - 부산 데이터] ⭐ 권장")
        ensemble_metrics = calculate_all_metrics(
            lstm_actuals_original, ensemble_predictions, print_details=True
        )
        
        # ==========================================
        # ⭐ 7. 미래 예측 생성 (24H, 48H, 72H) - 현재 날짜 기준
        # ==========================================
        predictions_df = generate_future_predictions(
            lstm_model, gru_model, scaler_X, scaler_y,
            X_test, df_valid, device
        )
        
        # ==========================================
        # ⭐ 8. 예측 결과 CSV 저장 (시간별 현재/누적 발전량 포함)
        # ==========================================
        print("\n" + "="*80)
        print("💾 예측 결과 CSV 저장 중...")
        print("="*80)
        
        saved_files = save_prediction_csvs(predictions_df, output_dir='./prediction_results')
        
        print("\n" + "="*80)
        print("📁 저장된 파일 목록:")
        print("="*80)
        for period, filepath in saved_files.items():
            print(f"  [{period:>3}] {filepath}")
        
        # 9. 최종 요약
        print("\n" + "="*80)
        print("📊 성능 비교 요약")
        print("="*80)
        
        print("\n대구 학습 성능 (원본 지역):")
        print(f"  LSTM  - R²: {metadata['lstm_metrics']['r2']:.4f}, MAPE: {metadata['lstm_metrics']['mape']:.2f}%")
        print(f"  GRU   - R²: {metadata['gru_metrics']['r2']:.4f}, MAPE: {metadata['gru_metrics']['mape']:.2f}%")
        
        print("\n부산 적용 성능 (전이 지역):")
        print(f"  LSTM      - R²: {lstm_metrics['r2']:.4f}, NMAE: {lstm_metrics['nmae']:.4f}, MAPE: {lstm_metrics['mape']:.2f}%")
        print(f"  GRU       - R²: {gru_metrics['r2']:.4f}, NMAE: {gru_metrics['nmae']:.4f}, MAPE: {gru_metrics['mape']:.2f}%")
        print(f"  앙상블    - R²: {ensemble_metrics['r2']:.4f}, NMAE: {ensemble_metrics['nmae']:.4f}, MAPE: {ensemble_metrics['mape']:.2f}%")
        
        # 10. 예측 샘플 출력
        print("\n" + "="*80)
        print("🔮 미래 예측 샘플 (처음 10시간)")
        print("="*80)
        print(predictions_df.head(10)[['예측_일시', '경과_시간(H)', '앙상블_현재_발전량(MWh)', '앙상블_누적_발전량(MWh)']].to_string(index=False))
        
        print("\n" + "="*80)
        print("🔮 미래 예측 요약")
        print("="*80)
        print(f"24H 후 누적 발전량: {predictions_df[predictions_df['경과_시간(H)'] == 24]['앙상블_누적_발전량(MWh)'].values[0]:.2f} MWh")
        print(f"48H 후 누적 발전량: {predictions_df[predictions_df['경과_시간(H)'] == 48]['앙상블_누적_발전량(MWh)'].values[0]:.2f} MWh")
        print(f"72H 후 누적 발전량: {predictions_df[predictions_df['경과_시간(H)'] == 72]['앙상블_누적_발전량(MWh)'].values[0]:.2f} MWh")
        
        print("\n" + "="*80)
        print("✅ 모든 작업 완료!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ 에러: {e}")
        print("\n확인 사항:")
        print("1. 대구 전이학습 모델이 ./saved_models/transfer_daegu 디렉토리에 있는지 확인")
        print("2. 부산 CSV 파일 경로가 올바른지 확인")
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()