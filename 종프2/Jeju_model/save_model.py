import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 터미널 환경을 위한 백엔드 설정
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import plot_importance



# GPU/CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 사용 중인 디바이스: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA 버전: {torch.version.cuda}")
    print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # GPU 메모리 최적화 설정
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 성능 향상
else:
    print("   ⚠️  CUDA를 사용할 수 없습니다. CPU를 사용합니다.")

# 데이터 경로
data_path = "./dataset/jeju_solar_utf8.csv"
warnings.filterwarnings("ignore")

# 결과 저장 디렉토리 생성
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Plot 저장 경로: {output_dir}")

# 모델 저장 디렉토리 생성
model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)
print(f"📁 모델 저장 경로: {model_dir}")

plt.style.use('seaborn-v0_8-whitegrid')

# 한글 폰트 설정 - 시스템에 맞는 폰트 자동 선택
import matplotlib.font_manager as fm
import platform

def set_korean_font():
    """
    시스템에 설치된 한글 폰트를 자동으로 찾아 설정
    """
    system = platform.system()
    
    # 우선순위가 높은 한글 폰트 목록
    korean_fonts = [
        'Malgun Gothic',      # Windows
        'AppleGothic',        # macOS
        'NanumGothic',        # 나눔고딕
        'NanumBarunGothic',   # 나눔바른고딕
        'Nanum Gothic',
        'DejaVu Sans'         # 기본 대체 폰트
    ]
    
    # 설치된 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 사용 가능한 한글 폰트 찾기
    for font in korean_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"✅ 한글 폰트 설정 완료: {font}")
            break
    else:
        # 한글 폰트가 없으면 기본 폰트 사용
        print("⚠️  한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        print("   한글이 깨질 수 있습니다. 나눔고딕 설치를 권장합니다.")
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 실행
set_korean_font()


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_mape(y_true, y_pred, method='improved'):
    """
    개선된 MAPE 계산 - 태양광 발전량 특성을 고려한 여러 방법 제공
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        method: 계산 방법
            - 'improved': 개선된 MAPE (기본값)
            - 'threshold': 임계값 기반 MAPE
            - 'weighted': 가중 MAPE
            - 'symmetric': 대칭 MAPE
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # NaN 및 무한값 확인 및 제거
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print("Warning: 유효한 데이터가 없어 MAPE를 계산할 수 없습니다.")
        return np.nan
    
    if method == 'improved':
        """
        개선된 MAPE: 작은 값들에 대한 가중치를 조정하여 
        전체적인 예측 성능을 더 잘 반영
        """
        # 전체 데이터의 분포를 고려한 임계값 설정
        threshold = np.percentile(y_true_clean, 10)  # 하위 10% 값을 임계값으로 사용
        
        # 임계값 이상인 데이터에 대해서만 MAPE 계산
        significant_mask = y_true_clean >= threshold
        
        if not np.any(significant_mask):
            # 모든 값이 임계값 미만인 경우, 절대 오차 기반 계산
            abs_errors = np.abs(y_true_clean - y_pred_clean)
            mean_actual = np.mean(y_true_clean)
            if mean_actual > 0:
                return (np.mean(abs_errors) / mean_actual) * 100
            else:
                return 0.0
        
        y_true_sig = y_true_clean[significant_mask]
        y_pred_sig = y_pred_clean[significant_mask]
        
        # 가중 평균 MAPE 계산
        weights = y_true_sig / np.sum(y_true_sig)  # 실제값에 비례한 가중치
        percentage_errors = np.abs((y_true_sig - y_pred_sig) / y_true_sig)
        
        # 극단적인 오차 제한
        percentage_errors = np.clip(percentage_errors, 0, 2)  # 최대 200% 오차로 제한
        
        mape_value = np.sum(weights * percentage_errors) * 100
        
        # 제거된 데이터 비율 출력
        removed_count = len(y_true_clean) - len(y_true_sig)
        if removed_count > 0:
            removal_rate = (removed_count / len(y_true_clean)) * 100
            print(f"MAPE 계산 시 작은 값 제외: {removed_count}개 ({removal_rate:.1f}%)")
    
    elif method == 'threshold':
        """
        임계값 기반 MAPE: 일정 값 이상의 데이터만 사용
        """
        # 동적 임계값 계산 (평균의 10%)
        threshold = np.mean(y_true_clean) * 0.1
        
        above_threshold = y_true_clean > threshold
        
        if not np.any(above_threshold):
            return 0.0
            
        y_true_filtered = y_true_clean[above_threshold]
        y_pred_filtered = y_pred_clean[above_threshold]
        
        percentage_errors = np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)
        percentage_errors = np.clip(percentage_errors, 0, 1.5)  # 150% 제한
        
        mape_value = np.mean(percentage_errors) * 100
        
        removed_count = len(y_true_clean) - len(y_true_filtered)
        print(f"임계값({threshold:.3f}) 미만 제외: {removed_count}개")
    
    elif method == 'weighted':
        """
        가중 MAPE: 값의 크기에 따라 가중치 부여
        """
        # 0에 가까운 값 제외
        non_zero_mask = y_true_clean > np.percentile(y_true_clean, 5)
        
        if not np.any(non_zero_mask):
            return 0.0
            
        y_true_nz = y_true_clean[non_zero_mask]
        y_pred_nz = y_pred_clean[non_zero_mask]
        
        # 실제값의 크기에 비례한 가중치
        weights = y_true_nz / np.sum(y_true_nz)
        
        percentage_errors = np.abs((y_true_nz - y_pred_nz) / y_true_nz)
        percentage_errors = np.clip(percentage_errors, 0, 1.0)  # 100% 제한
        
        mape_value = np.sum(weights * percentage_errors) * 100
    
    elif method == 'symmetric':
        """
        대칭 MAPE (SMAPE): 분모에 실제값과 예측값의 평균 사용
        """
        # 매우 작은 값들 제외
        min_threshold = np.percentile(np.abs(y_true_clean), 5)
        valid_mask = np.abs(y_true_clean) > min_threshold
        
        if not np.any(valid_mask):
            return 0.0
            
        y_true_filtered = y_true_clean[valid_mask]
        y_pred_filtered = y_pred_clean[valid_mask]
        
        denominator = (np.abs(y_true_filtered) + np.abs(y_pred_filtered)) / 2
        percentage_errors = np.abs(y_true_filtered - y_pred_filtered) / denominator
        percentage_errors = np.clip(percentage_errors, 0, 1.0)  # 100% 제한
        
        mape_value = np.mean(percentage_errors) * 100
    
    else:
        raise ValueError("지원하지 않는 MAPE 계산 방법입니다.")
    
    return mape_value

def calculate_normalized_mape(y_true, y_pred):
    """
    정규화된 MAPE: 데이터 범위에 따라 정규화
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # 유효한 데이터만 선택
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        print("Warning: 유효한 데이터가 없어 정규화된 MAPE를 계산할 수 없습니다.")
        return np.nan
    
    # 데이터 범위 계산
    data_range = np.max(y_true_valid) - np.min(y_true_valid)
    
    if data_range == 0:
        return 0.0
    
    # 절대 오차의 평균을 데이터 범위로 정규화
    abs_errors = np.abs(y_true_valid - y_pred_valid)
    normalized_mape = (np.mean(abs_errors) / data_range) * 100
    
    return normalized_mape


def calculate_all_metrics(y_true, y_pred, print_details=False):
    """
    모든 평가 지표를 한 번에 계산하고 반환
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'r2': calculate_r2(y_true, y_pred),
        'mape_improved': calculate_mape(y_true, y_pred, method='improved'),
        'mape_threshold': calculate_mape(y_true, y_pred, method='threshold'),
        'mape_weighted': calculate_mape(y_true, y_pred, method='weighted'),
        'mape_symmetric': calculate_mape(y_true, y_pred, method='symmetric'),
        'normalized_mape': calculate_normalized_mape(y_true, y_pred),
    }
    
    # 데이터 범위 계산
    data_range = np.max(y_true) - np.min(y_true)
    
    # NMAE 및 NRMSE 계산
    metrics['nmae'] = metrics['mae'] / data_range if data_range > 0 else 0
    metrics['nrmse'] = metrics['rmse'] / data_range if data_range > 0 else 0
    
    if print_details:
        print("\n=== 상세 평가 지표 ===")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"NMAE: {metrics['nmae']:.4f}")
        print(f"NRMSE: {metrics['nrmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"\n=== MAPE 변형 지표 ===")
        print(f"개선된 MAPE: {metrics['mape_improved']:.2f}%")
        print(f"임계값 MAPE: {metrics['mape_threshold']:.2f}%")
        print(f"가중 MAPE: {metrics['mape_weighted']:.2f}%")
        print(f"대칭 MAPE: {metrics['mape_symmetric']:.2f}%")
        print(f"정규화 MAPE: {metrics['normalized_mape']:.2f}%")
    
    return metrics


# === 데이터 로딩 및 전처리 함수 ===
def load_and_preprocess_data(data_path, sequence_length=24):
    """
    데이터 로딩 및 전처리
    """
    print("\n데이터 로딩 중...")
    df = pd.read_csv(data_path)
    print(f"원본 데이터 크기: {df.shape}")
    
    # 결측값 처리
    print("\n결측값 처리 중...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # 타겟 변수 선택
    target_col = '태양광 발전량(MWh)'
    if target_col not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_col}'을 찾을 수 없습니다.")
    
    # 특성과 타겟 분리
    feature_cols = [col for col in numeric_columns if col != target_col]
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    print(f"특성 개수: {len(feature_cols)}")
    print(f"특성 목록: {feature_cols}")
    
    # 데이터 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 시퀀스 데이터 생성
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_scaled[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"\n시퀀스 데이터 shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Train/Val/Test 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    print(f"\n데이터 분할:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            scaler_X, scaler_y, feature_cols)


# === PyTorch Dataset 클래스 ===
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
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward
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
            input_size, 
            hidden_size, 
            num_layers, 
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


# === 학습 함수 ===
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=100, patience=15, device='cpu'):
    """
    모델 학습 함수
    """
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n모델 학습 시작 (총 {num_epochs} 에폭)")
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
        
        # 진행상황 출력
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
    
    # 최적 모델 로드
    model.load_state_dict(best_model_state)
    
    elapsed_time = time.time() - start_time
    print(f"학습 완료! 소요 시간: {elapsed_time:.2f}초")
    
    return model, train_losses, val_losses


# === 예측 함수 ===
def predict(model, test_loader, device='cpu'):
    """
    모델 예측 함수
    """
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


# === 모델 저장 함수 ===
def save_models(lstm_model, gru_model, xgb_model, scaler_X, scaler_y, 
                feature_cols, lstm_metrics, gru_metrics, stacked_metrics, 
                model_dir='./saved_models'):
    """
    학습된 모델과 스케일러, 메타데이터 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print("모델 저장 중...")
    print(f"{'='*80}")
    
    # 1. PyTorch 모델 저장 (LSTM)
    lstm_path = os.path.join(model_dir, f'lstm_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': lstm_model.state_dict(),
        'model_config': {
            'input_size': lstm_model.lstm.input_size,
            'hidden_size': lstm_model.hidden_size,
            'num_layers': lstm_model.num_layers,
        },
        'metrics': lstm_metrics
    }, lstm_path)
    print(f"✅ LSTM 모델 저장: {lstm_path}")
    
    # 2. PyTorch 모델 저장 (GRU)
    gru_path = os.path.join(model_dir, f'gru_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': gru_model.state_dict(),
        'model_config': {
            'input_size': gru_model.gru.input_size,
            'hidden_size': gru_model.hidden_size,
            'num_layers': gru_model.num_layers,
        },
        'metrics': gru_metrics
    }, gru_path)
    print(f"✅ GRU 모델 저장: {gru_path}")
    
    # 3. XGBoost 모델 저장
    xgb_path = os.path.join(model_dir, f'xgboost_stacking_{timestamp}.json')
    xgb_model.save_model(xgb_path)
    print(f"✅ XGBoost 스태킹 모델 저장: {xgb_path}")
    
    # 4. 스케일러 저장
    scaler_path = os.path.join(model_dir, f'scalers_{timestamp}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }, f)
    print(f"✅ 스케일러 저장: {scaler_path}")
    
    # 5. 메타데이터 저장 (특성 정보, 성능 지표 등)
    metadata = {
        'timestamp': timestamp,
        'feature_columns': feature_cols,
        'sequence_length': 24,
        'device': str(device),
        'lstm_metrics': {k: float(v) if not isinstance(v, str) else v 
                        for k, v in lstm_metrics.items()},
        'gru_metrics': {k: float(v) if not isinstance(v, str) else v 
                       for k, v in gru_metrics.items()},
        'stacked_metrics': {k: float(v) if not isinstance(v, str) else v 
                           for k, v in stacked_metrics.items()},
        'model_files': {
            'lstm': f'lstm_model_{timestamp}.pth',
            'gru': f'gru_model_{timestamp}.pth',
            'xgboost': f'xgboost_stacking_{timestamp}.json',
            'scalers': f'scalers_{timestamp}.pkl'
        }
    }
    
    metadata_path = os.path.join(model_dir, f'metadata_{timestamp}.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"✅ 메타데이터 저장: {metadata_path}")
    
    # 6. 최신 모델 경로를 가리키는 링크 파일 생성
    latest_models_info = {
        'timestamp': timestamp,
        'lstm_model': lstm_path,
        'gru_model': gru_path,
        'xgboost_model': xgb_path,
        'scalers': scaler_path,
        'metadata': metadata_path
    }
    
    latest_path = os.path.join(model_dir, 'latest_models.json')
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(latest_models_info, f, indent=4, ensure_ascii=False)
    print(f"✅ 최신 모델 정보 저장: {latest_path}")
    
    print(f"\n{'='*80}")
    print(f"✨ 모든 모델이 성공적으로 저장되었습니다!")
    print(f"{'='*80}")
    
    return {
        'lstm': lstm_path,
        'gru': gru_path,
        'xgboost': xgb_path,
        'scalers': scaler_path,
        'metadata': metadata_path
    }


# === 모델 로드 함수 ===
def load_models(model_dir='./saved_models', timestamp=None):
    """
    저장된 모델과 스케일러 로드
    
    Args:
        model_dir: 모델이 저장된 디렉토리
        timestamp: 특정 시점의 모델을 로드하려면 타임스탬프 지정
                  None이면 최신 모델 로드
    """
    print(f"\n{'='*80}")
    print("모델 로드 중...")
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
    
    # 스케일러 로드
    scaler_path = os.path.join(model_dir, f'scalers_{timestamp}.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']
    print(f"✅ 스케일러 로드: {scaler_path}")
    
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
    lstm_model.eval()
    print(f"✅ LSTM 모델 로드: {lstm_path}")
    
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
    gru_model.eval()
    print(f"✅ GRU 모델 로드: {gru_path}")
    
    # XGBoost 모델 로드
    xgb_path = os.path.join(model_dir, f'xgboost_stacking_{timestamp}.json')
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    print(f"✅ XGBoost 스태킹 모델 로드: {xgb_path}")
    
    print(f"\n{'='*80}")
    print(f"✨ 모든 모델이 성공적으로 로드되었습니다!")
    print(f"{'='*80}")
    
    return {
        'lstm_model': lstm_model,
        'gru_model': gru_model,
        'xgb_model': xgb_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metadata': metadata
    }


# === 메인 실행 코드 ===
if __name__ == "__main__":
    try:
        # 1. 데이터 로딩 및 전처리
        SEQUENCE_LENGTH = 24
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         scaler_X, scaler_y, feature_cols) = load_and_preprocess_data(data_path, SEQUENCE_LENGTH)
        
        # 2. DataLoader 생성
        BATCH_SIZE = 64
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # 3. LSTM 모델 학습
        print("\n" + "="*80)
        print("LSTM 모델 학습")
        print("="*80)
        
        input_size = X_train.shape[2]
        lstm_model = LSTMModel(input_size=input_size, hidden_size=128, 
                              num_layers=2, dropout=0.2).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        
        lstm_model, lstm_train_losses, lstm_val_losses = train_model(
            lstm_model, train_loader, val_loader, criterion, optimizer,
            num_epochs=100, patience=15, device=device
        )
        
        # LSTM 예측
        lstm_predictions, lstm_actuals = predict(lstm_model, test_loader, device)
        
        # 원래 스케일로 복원
        lstm_predictions_original = scaler_y.inverse_transform(lstm_predictions)
        lstm_actuals_original = scaler_y.inverse_transform(lstm_actuals)
        
        # LSTM 평가
        lstm_metrics = calculate_all_metrics(lstm_actuals_original, lstm_predictions_original, print_details=True)
        nmae_lstm = lstm_metrics['nmae']
        nrmse_lstm = lstm_metrics['nrmse']
        r2_lstm = lstm_metrics['r2']
        mape_lstm = lstm_metrics['mape_improved']
        
        print(f"\n=== LSTM 모델 성능 평가 ===")
        print(f"NMAE: {nmae_lstm:.4f}")
        print(f"NRMSE: {nrmse_lstm:.4f}")
        print(f"R²: {r2_lstm:.4f}")
        print(f"MAPE: {mape_lstm:.4f}%")
        
        # 4. GRU 모델 학습
        print("\n" + "="*80)
        print("GRU 모델 학습")
        print("="*80)
        
        gru_model = GRUModel(input_size=input_size, hidden_size=128, 
                            num_layers=2, dropout=0.2).to(device)
        optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
        
        gru_model, gru_train_losses, gru_val_losses = train_model(
            gru_model, train_loader, val_loader, criterion, optimizer,
            num_epochs=100, patience=15, device=device
        )
        
        # GRU 예측
        gru_predictions, gru_actuals = predict(gru_model, test_loader, device)
        
        # 원래 스케일로 복원
        gru_predictions_original = scaler_y.inverse_transform(gru_predictions)
        gru_actuals_original = scaler_y.inverse_transform(gru_actuals)
        
        # GRU 평가
        gru_metrics = calculate_all_metrics(gru_actuals_original, gru_predictions_original, print_details=True)
        nmae_gru = gru_metrics['nmae']
        nrmse_gru = gru_metrics['nrmse']
        r2_gru = gru_metrics['r2']
        mape_gru = gru_metrics['mape_improved']
        
        print(f"\n=== GRU 모델 성능 평가 ===")
        print(f"NMAE: {nmae_gru:.4f}")
        print(f"NRMSE: {nrmse_gru:.4f}")
        print(f"R²: {r2_gru:.4f}")
        print(f"MAPE: {mape_gru:.4f}%")
        
        # 5. 스태킹을 위한 메타 특성 생성
        print("\n" + "="*80)
        print("스태킹 모델 준비")
        print("="*80)
        
        # Test set에 대한 LSTM과 GRU의 예측값을 특성으로 사용
        X_test_stack = np.hstack([
            lstm_predictions_original.reshape(-1, 1),
            gru_predictions_original.reshape(-1, 1)
        ])
        y_test_stack = lstm_actuals_original.flatten()
        
        # Train set에서도 동일하게 메타 특성 생성
        lstm_train_predictions, _ = predict(lstm_model, train_loader, device)
        gru_train_predictions, _ = predict(gru_model, train_loader, device)
        
        lstm_train_predictions_original = scaler_y.inverse_transform(lstm_train_predictions)
        gru_train_predictions_original = scaler_y.inverse_transform(gru_train_predictions)
        
        X_train_stack = np.hstack([
            lstm_train_predictions_original.reshape(-1, 1),
            gru_train_predictions_original.reshape(-1, 1)
        ])
        y_train_stack = scaler_y.inverse_transform(
            train_dataset.y.numpy()
        ).flatten()
        
        # Validation set
        lstm_val_predictions, _ = predict(lstm_model, val_loader, device)
        gru_val_predictions, _ = predict(gru_model, val_loader, device)
        
        lstm_val_predictions_original = scaler_y.inverse_transform(lstm_val_predictions)
        gru_val_predictions_original = scaler_y.inverse_transform(gru_val_predictions)
        
        X_val_stack = np.hstack([
            lstm_val_predictions_original.reshape(-1, 1),
            gru_val_predictions_original.reshape(-1, 1)
        ])
        y_val_stack = scaler_y.inverse_transform(
            val_dataset.y.numpy()
        ).flatten()
        
        print(f"스태킹 데이터 크기:")
        print(f"  Train: {X_train_stack.shape}")
        print(f"  Val: {X_val_stack.shape}")
        print(f"  Test: {X_test_stack.shape}")
        
        # 6. XGBoost 스태킹 모델 학습
        print("\n" + "="*80)
        print("XGBoost 스태킹 모델 학습")
        print("="*80)
        
        feature_names = ['LSTM_prediction', 'GRU_prediction']
        
        xgb_stacking_regressor = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            eval_metric='mae'
        )
        
        print("XGBoost 스태킹 모델 학습 중...")
        xgb_stacking_regressor.fit(
            X_train_stack, 
            y_train_stack, 
            eval_set=[(X_val_stack, y_val_stack)], 
            verbose=100
        )
        
        # 스태킹 모델 예측 및 평가
        stacked_pred_test = xgb_stacking_regressor.predict(X_test_stack)

        # 스태킹 모델 평가
        stacked_metrics = calculate_all_metrics(y_test_stack, stacked_pred_test, print_details=True)
        nmae_stacked = stacked_metrics['nmae']
        nrmse_stacked = stacked_metrics['nrmse']
        r2_stacked = stacked_metrics['r2']
        mape_stacked = stacked_metrics['mape_improved']
        
        print(f"\n=== XGBoost 스태킹 모델 성능 평가 ===")
        print(f"NMAE: {nmae_stacked:.4f}")
        print(f"NRMSE: {nrmse_stacked:.4f}")
        print(f"R²: {r2_stacked:.4f}")
        print(f"MAPE: {mape_stacked:.4f}%")
        
        # 7. 특성 중요도 분석
        print("\n=== XGBoost 스태킹 모델 특성 중요도 ===")
        importance_dict = dict(zip(feature_names, xgb_stacking_regressor.feature_importances_))
        for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # 8. 최종 성능 비교
        print(f"\n{'='*80}")
        print("최종 모델 성능 비교")
        print(f"{'='*80}")
        print(f"{'모델':<20} {'NMAE':<10} {'NRMSE':<10} {'R²':<10} {'MAPE':<10}")
        print("-" * 80)
        print(f"{'LSTM':<20} {nmae_lstm:<10.4f} {nrmse_lstm:<10.4f} {r2_lstm:<10.4f} {mape_lstm:<10.2f}%")
        print(f"{'GRU':<20} {nmae_gru:<10.4f} {nrmse_gru:<10.4f} {r2_gru:<10.4f} {mape_gru:<10.2f}%")
        print(f"{'XGBoost Stacking':<20} {nmae_stacked:<10.4f} {nrmse_stacked:<10.4f} {r2_stacked:<10.4f} {mape_stacked:<10.2f}%")
        
        # 성능 개선율 계산
        best_individual = min(nmae_lstm, nmae_gru)
        improvement = ((best_individual - nmae_stacked) / best_individual) * 100
        print(f"\n스태킹으로 인한 성능 개선: {improvement:.2f}%")
        
        # ========================================
        # 9. 모델 저장 (새로 추가된 부분)
        # ========================================
        saved_paths = save_models(
            lstm_model=lstm_model,
            gru_model=gru_model,
            xgb_model=xgb_stacking_regressor,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            feature_cols=feature_cols,
            lstm_metrics=lstm_metrics,
            gru_metrics=gru_metrics,
            stacked_metrics=stacked_metrics,
            model_dir=model_dir
        )
        
        # 10. 결과 시각화 (파일로 저장)
        print("\n결과 시각화 중...")
        
        # 첫 번째 그림: 모델별 상세 분석
        fig1 = plt.figure(figsize=(20, 15))
        
        # 1. 학습 곡선
        plt.subplot(3, 4, 1)
        if lstm_train_losses and lstm_val_losses:
            plt.plot(lstm_train_losses, label='LSTM Train Loss', alpha=0.7)
            plt.plot(lstm_val_losses, label='LSTM Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 2)
        if gru_train_losses and gru_val_losses:
            plt.plot(gru_train_losses, label='GRU Train Loss', alpha=0.7)
            plt.plot(gru_val_losses, label='GRU Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GRU Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 예측 결과 비교
        test_range = min(200, len(y_test_stack))
        
        plt.subplot(3, 4, 3)
        plt.plot(lstm_actuals_original[:test_range], label='Actual', alpha=0.8, linewidth=2)
        plt.plot(lstm_predictions_original[:test_range], label='LSTM', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('발전량 (MWh)')
        plt.title(f'LSTM 예측 결과\nNMAE: {nmae_lstm:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 4)
        plt.plot(gru_actuals_original[:test_range], label='Actual', alpha=0.8, linewidth=2)
        plt.plot(gru_predictions_original[:test_range], label='GRU', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('발전량 (MWh)')
        plt.title(f'GRU 예측 결과\nNMAE: {nmae_gru:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 5)
        plt.plot(y_test_stack[:test_range], label='Actual', alpha=0.8, linewidth=2)
        plt.plot(stacked_pred_test[:test_range], label='Stacked', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('발전량 (MWh)')
        plt.title(f'XGBoost 스태킹 결과\nNMAE: {nmae_stacked:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 산점도 (실제값 vs 예측값)
        plt.subplot(3, 4, 6)
        plt.scatter(lstm_actuals_original, lstm_predictions_original, alpha=0.5)
        plt.plot([lstm_actuals_original.min(), lstm_actuals_original.max()], 
                [lstm_actuals_original.min(), lstm_actuals_original.max()], 'r--', lw=2)
        plt.xlabel('실제값')
        plt.ylabel('예측값')
        plt.title('LSTM: 실제값 vs 예측값')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 7)
        plt.scatter(gru_actuals_original, gru_predictions_original, alpha=0.5)
        plt.plot([gru_actuals_original.min(), gru_actuals_original.max()], 
                [gru_actuals_original.min(), gru_actuals_original.max()], 'r--', lw=2)
        plt.xlabel('실제값')
        plt.ylabel('예측값')
        plt.title('GRU: 실제값 vs 예측값')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 8)
        plt.scatter(y_test_stack, stacked_pred_test, alpha=0.5)
        plt.plot([y_test_stack.min(), y_test_stack.max()], 
                [y_test_stack.min(), y_test_stack.max()], 'r--', lw=2)
        plt.xlabel('실제값')
        plt.ylabel('예측값')
        plt.title('Stacked: 실제값 vs 예측값')
        plt.grid(True, alpha=0.3)
        
        # 4. 특성 중요도
        plt.subplot(3, 4, 9)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_stacking_regressor.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('중요도')
        plt.title('XGBoost 스태킹 특성 중요도')
        plt.grid(True, alpha=0.3)
        
        # 5. 잔차 분석
        lstm_residuals = lstm_actuals_original - lstm_predictions_original
        gru_residuals = gru_actuals_original - gru_predictions_original
        stacked_residuals = y_test_stack - stacked_pred_test
        
        plt.subplot(3, 4, 10)
        plt.scatter(lstm_predictions_original, lstm_residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값')
        plt.ylabel('잔차')
        plt.title('LSTM 잔차 플롯')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 11)
        plt.scatter(gru_predictions_original, gru_residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값')
        plt.ylabel('잔차')
        plt.title('GRU 잔차 플롯')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 12)
        plt.scatter(stacked_pred_test, stacked_residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값')
        plt.ylabel('잔차')
        plt.title('Stacked 잔차 플롯')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, '01_detailed_analysis.png')
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"✅ 상세 분석 그래프 저장: {plot1_path}")
        plt.close()
        
        # 11. 성능 메트릭 비교 막대 그래프
        fig2 = plt.figure(figsize=(15, 5))
        
        models = ['LSTM', 'GRU', 'XGBoost\nStacking']
        nmae_scores = [nmae_lstm, nmae_gru, nmae_stacked]
        nrmse_scores = [nrmse_lstm, nrmse_gru, nrmse_stacked]
        r2_scores = [r2_lstm, r2_gru, r2_stacked]
        mape_scores = [mape_lstm, mape_gru, mape_stacked]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.subplot(1, 4, 1)
        plt.bar(x, nmae_scores, width, label='NMAE', alpha=0.8)
        plt.xlabel('모델')
        plt.ylabel('NMAE')
        plt.title('NMAE 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 4, 2)
        plt.bar(x, nrmse_scores, width, label='NRMSE', alpha=0.8, color='orange')
        plt.xlabel('모델')
        plt.ylabel('NRMSE')
        plt.title('NRMSE 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 3)
        plt.bar(x, r2_scores, width, label='R²', alpha=0.8, color='red')
        plt.xlabel('모델')
        plt.ylabel('R²')
        plt.title('R² 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 4)
        plt.bar(x, mape_scores, width, label='MAPE (%)', alpha=0.8, color='green')
        plt.xlabel('모델')
        plt.ylabel('MAPE (%)')
        plt.title('MAPE 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, '02_metrics_comparison.png')
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"✅ 성능 비교 그래프 저장: {plot2_path}")
        plt.close()
        
        print(f"\n{'='*80}")
        print("모델 학습 및 평가 완료!")
        print(f"{'='*80}")
        print(f"\n📊 모든 그래프가 '{output_dir}' 폴더에 저장되었습니다:")
        print(f"  - {plot1_path}")
        print(f"  - {plot2_path}")
        
        print(f"\n💾 저장된 모델 파일:")
        for model_type, path in saved_paths.items():
            print(f"  - {model_type}: {path}")
        
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()