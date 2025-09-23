import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
data_path = "C:/Users/rlask/종프2/dataset/jeju_solar_utf8.csv"
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-whitegrid')
# 한글 폰트 설정 (Windows 사용자용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    # Handle cases where all true values are zero
    if not np.any(non_zero_mask):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def preprocess_data(data_df):
    """
    데이터 전처리 함수 - 결측값 처리 및 데이터 클리닝
    """
    print("데이터 전처리 시작...")
    
    # 데이터 정보 출력
    print(f"원본 데이터 크기: {data_df.shape}")
    print(f"결측값 개수:")
    missing_counts = data_df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(data_df)*100:.1f}%)")
    
    # 필요한 컬럼만 선택
    required_columns = ["기온", "강수량(mm)", "일조(hr)", "일사량", "태양광 발전량(MWh)"]
    
    # 컬럼 존재 여부 확인
    missing_cols = [col for col in required_columns if col not in data_df.columns]
    if missing_cols:
        print(f"누락된 컬럼: {missing_cols}")
        return None, None
    
    # 데이터 추출
    processed_df = data_df[required_columns].copy()
    
    # 결측값 처리
    # 1. 기온: 전후 값의 평균으로 보간
    processed_df['기온'] = processed_df['기온'].interpolate(method='linear')
    
    # 2. 강수량: 0으로 채움 (비가 안 온 것으로 가정)
    processed_df['강수량(mm)'] = processed_df['강수량(mm)'].fillna(0)
    
    # 3. 일조와 일사량: 계절성과 시간대를 고려한 보간
    if '일시' in data_df.columns:
        # 날짜 정보가 있다면 시간대별 평균으로 보간
        data_df['datetime'] = pd.to_datetime(data_df['일시'])
        data_df['hour'] = data_df['datetime'].dt.hour
        data_df['month'] = data_df['datetime'].dt.month
        
        # 시간대별, 월별 평균값으로 결측값 보간
        for col in ['일조(hr)', '일사량']:
            if col in processed_df.columns:
                # 먼저 시간대별 평균으로 보간
                hourly_mean = data_df.groupby('hour')[col].transform('mean')
                processed_df[col] = processed_df[col].fillna(hourly_mean)
                
                # 여전히 NaN이 있다면 전체 평균으로 보간
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    else:
        # 날짜 정보가 없다면 단순 보간
        processed_df['일조(hr)'] = processed_df['일조(hr)'].interpolate(method='linear')
        processed_df['일사량'] = processed_df['일사량'].interpolate(method='linear')
        
        # 여전히 NaN이 있다면 평균값으로 채움
        processed_df['일조(hr)'] = processed_df['일조(hr)'].fillna(processed_df['일조(hr)'].mean())
        processed_df['일사량'] = processed_df['일사량'].fillna(processed_df['일사량'].mean())
    
    # 4. 발전량: 0으로 채움 (발전이 안 된 것으로 가정)
    processed_df['태양광 발전량(MWh)'] = processed_df['태양광 발전량(MWh)'].fillna(0)
    
    # 이상값 제거 (IQR 방법 사용)
    def remove_outliers(series, factor=1.5):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return series.clip(lower, upper)
    
    # 기온과 발전량에 대해서만 이상값 처리
    processed_df['기온'] = remove_outliers(processed_df['기온'])
    processed_df['태양광 발전량(MWh)'] = remove_outliers(processed_df['태양광 발전량(MWh)'])
    
    # 최종 결측값 확인
    final_missing = processed_df.isnull().sum()
    if final_missing.sum() > 0:
        print("전처리 후 남은 결측값:")
        for col, count in final_missing.items():
            if count > 0:
                print(f"  {col}: {count}")
        
        # 남은 결측값이 있다면 해당 행 제거
        processed_df = processed_df.dropna()
        print(f"결측값이 있는 행 제거 후 데이터 크기: {processed_df.shape}")
    
    # 특징과 타겟 분리
    features = processed_df[["기온", "강수량(mm)", "일조(hr)", "일사량"]].values
    targets = processed_df["태양광 발전량(MWh)"].values
    
    print(f"전처리 완료 - 최종 데이터 크기: {len(features)} 행")
    print(f"특징 데이터 형태: {features.shape}")
    print(f"타겟 데이터 형태: {targets.shape}")
    
    return features, targets


class PatternExtractor:
    """
    기상 데이터에서 태양광 발전량과 관련된 패턴을 추출하는 클래스
    """
    def __init__(self, n_patterns=5):
        self.n_patterns = n_patterns
        self.kmeans = None
        self.pattern_labels = None
        self.pattern_centers = None
        self.imputer = SimpleImputer(strategy='mean')  # 추가 안전장치
        
    def extract_weather_patterns(self, features, targets):
        """
        기상 조건과 발전량의 관계에서 패턴 추출
        """
        # 입력 데이터 검증 및 결측값 처리
        print("패턴 추출을 위한 데이터 검증 중...")
        
        # NaN 체크
        if np.isnan(features).any():
            print("Warning: features에 NaN 값이 발견되어 imputer로 처리합니다.")
            features = self.imputer.fit_transform(features)
        
        if np.isnan(targets).any():
            print("Warning: targets에 NaN 값이 발견되어 평균값으로 처리합니다.")
            targets = np.nan_to_num(targets, nan=np.nanmean(targets))
        
        # 무한값 체크 및 처리
        if np.isinf(features).any():
            print("Warning: features에 무한값이 발견되어 처리합니다.")
            features = np.nan_to_num(features, posinf=np.nanmax(features[np.isfinite(features)]), 
                                     neginf=np.nanmin(features[np.isfinite(features)]))
        
        if np.isinf(targets).any():
            print("Warning: targets에 무한값이 발견되어 처리합니다.")
            targets = np.nan_to_num(targets, posinf=np.nanmax(targets[np.isfinite(targets)]), 
                                     neginf=np.nanmin(targets[np.isfinite(targets)]))
        
        # 기상 데이터와 발전량을 결합하여 패턴 분석
        combined_data = np.column_stack([features, targets.reshape(-1, 1)])
        
        # 데이터 정규화 (클러스터링 성능 향상을 위해)
        scaler = StandardScaler()
        combined_data_scaled = scaler.fit_transform(combined_data)
        
        # K-means 클러스터링으로 패턴 추출
        self.kmeans = KMeans(n_clusters=self.n_patterns, random_state=42, n_init=10)
        self.pattern_labels = self.kmeans.fit_predict(combined_data_scaled)
        
        # 원본 스케일의 센터 계산 (해석을 위해)
        self.pattern_centers = []
        for i in range(self.n_patterns):
            mask = self.pattern_labels == i
            if mask.sum() > 0:
                center = combined_data[mask].mean(axis=0)
                self.pattern_centers.append(center)
            else:
                # 빈 클러스터인 경우 전체 평균 사용
                self.pattern_centers.append(combined_data.mean(axis=0))
        
        self.pattern_centers = np.array(self.pattern_centers)
        
        print(f"추출된 패턴 수: {self.n_patterns}")
        self._analyze_patterns()
        
        return self.pattern_labels
    
    def _analyze_patterns(self):
        """
        추출된 패턴 분석 및 출력
        """
        print("\n=== 패턴 분석 결과 ===")
        pattern_names = ["저발전", "저중발전", "중발전", "중고발전", "고발전"]
        
        # 발전량 기준으로 패턴 정렬
        pattern_power = [(i, center[4]) for i, center in enumerate(self.pattern_centers)]
        pattern_power.sort(key=lambda x: x[1])
        
        for idx, (pattern_idx, power) in enumerate(pattern_power):
            center = self.pattern_centers[pattern_idx]
            pattern_name = pattern_names[idx] if idx < len(pattern_names) else f"패턴{idx}"
            count = (self.pattern_labels == pattern_idx).sum()
            
            print(f"패턴 {pattern_idx+1} ({pattern_name}) - 데이터 수: {count}")
            print(f"  - 평균 기온: {center[0]:.2f}°C")
            print(f"  - 평균 강수량: {center[1]:.2f}mm")
            print(f"  - 평균 일조시간: {center[2]:.2f}hr")
            print(f"  - 평균 일사량: {center[3]:.2f}")
            print(f"  - 평균 발전량: {center[4]:.2f}MWh")
            print()
    
    def get_pattern_features(self, features):
        """
        새로운 데이터에 대해 패턴 레이블 예측
        """
        if self.kmeans is None:
            raise ValueError("먼저 extract_weather_patterns를 실행해주세요.")
        
        # NaN 처리
        if np.isnan(features).any():
            features = self.imputer.transform(features)
        
        # 발전량이 없는 경우, 기상 데이터만으로 패턴 예측
        dummy_targets = np.zeros((len(features), 1))
        combined_data = np.column_stack([features, dummy_targets])
        
        return self.kmeans.predict(combined_data)


class SolarPatternDataset(Dataset):
    """
    패턴 정보를 포함한 태양광 발전 데이터셋
    """
    def __init__(self, features, targets, patterns, seq_len=24):
        self.X, self.y, self.patterns = [], [], []
        
        # patterns는 1D 배열 (정수 레이블) 이라고 가정
        patterns = np.array(patterns).astype(np.int64)
        
        # 시퀀스 데이터 생성 (패턴 정보 포함)
        for i in range(len(features) - seq_len):
            self.X.append(features[i:i+seq_len])
            self.y.append(targets[i+seq_len])
            self.patterns.append(patterns[i:i+seq_len])  # 패턴 시퀀스

        self.X = np.array(self.X, dtype=np.float32)      # (N, seq_len, feature_dim)
        self.y = np.array(self.y, dtype=np.float32)
        self.patterns = np.array(self.patterns, dtype=np.int64)  # 정수로 저장

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 기상 데이터와 패턴 정보를 결합
        # 패턴은 정수이므로 float로 변환하여 concat
        pattern_seq = self.patterns[idx].reshape(-1, 1).astype(np.float32)
        features_with_patterns = np.concatenate([
            self.X[idx], 
            pattern_seq
        ], axis=1)
        
        return torch.tensor(features_with_patterns), torch.tensor(self.y[idx])



class LSTM_Pattern(nn.Module):
    """
    패턴 정보를 활용한 LSTM 모델 (CNN 제거)
    """
    def __init__(self, weather_feature_dim=4, pattern_embedding_dim=8, hidden_dim=64, num_layers=2, n_patterns=5):
        super(LSTM_Pattern, self).__init__()
        
        self.n_patterns = n_patterns
        
        # Pattern Embedding Layer
        self.pattern_embed = nn.Embedding(n_patterns, pattern_embedding_dim)
        
        # LSTM (기상 데이터 + 패턴 임베딩)
        lstm_input_size = weather_feature_dim + pattern_embedding_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # FC layers
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        batch_size, seq_len, total_features = x.shape
        
        # 기상 데이터와 패턴 데이터 분리
        weather_data = x[:, :, :4]  # 기온, 강수량, 일조, 일사량
        pattern_data = x[:, :, 4].long()  # 패턴 레이블
        
        # Pattern embedding
        pattern_emb = self.pattern_embed(pattern_data)  # (batch, seq_len, embedding_dim)
        
        # Combine weather data with pattern embedding
        combined_features = torch.cat([weather_data, pattern_emb], dim=2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state
        out = attn_out[:, -1, :]  # (batch, hidden_dim)
        
        # FC layers with batch normalization
        out = self.fc1(out)
        out = self.batch_norm1(out)
        # Changed self.relu(out) to F.relu(out)
        out = F.relu(out)
        # --- FIX END ---
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        # --- FIX START ---
        # Changed self.relu(out) to F.relu(out)
        out = F.relu(out)
        # --- FIX END ---
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out.squeeze()
    
    def train_model(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """
        모델 학습 함수
        """
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        val_losses = []
        
        print(f"총 {epochs} 에포크 학습을 시작합니다...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                # CUDA로 이동
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = self(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        # CUDA로 이동
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        preds = self(batch_X)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                scheduler.step()
                
                # 진행상황 출력
                epoch_time = time.time() - epoch_start_time
                elapsed_time = time.time() - start_time
                remaining_epochs = epochs - epoch - 1
                estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs
                
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                      f"시간: {epoch_time:.1f}s | "
                      f"예상 남은 시간: {estimated_remaining/60:.1f}분")
        
        total_time = time.time() - start_time
        print(f"\n학습 완료! 총 소요시간: {total_time/60:.1f}분")
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
            """
            모델 예측 함수
            """
            self.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    # CUDA로 이동
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    preds = self(batch_X)
                    
                    # Convert tensors to numpy arrays
                    preds_np = preds.cpu().numpy()
                    actuals_np = batch_y.cpu().numpy()
                    
                    # Ensure the numpy arrays are always at least 1-dimensional.
                    # This prevents the "iteration over a 0-d array" error for single-item batches.
                    if preds_np.ndim == 0:
                        predictions.append(preds_np.item())
                        actuals.append(actuals_np.item())
                    else:
                        predictions.extend(preds_np)
                        actuals.extend(actuals_np)
            
            return np.array(predictions), np.array(actuals)

### <<< MODIFIED SECTION END >>>

def create_pattern_features(features, targets, pattern_extractor):
    """
    패턴 기반 특징 생성
    """
    # 기본 패턴 추출
    patterns = pattern_extractor.extract_weather_patterns(features, targets)
    
    # 추가 패턴 기반 특징 생성
    pattern_features = []
    
    for i in range(len(features)):
        pattern_feat = [
            patterns[i],  # 현재 패턴
            # 계절성 특징
            np.sin(2 * np.pi * i / 365),  # 연간 주기
            np.cos(2 * np.pi * i / 365),
            np.sin(2 * np.pi * i / 24),   # 일간 주기 (가정)
            np.cos(2 * np.pi * i / 24),
            # 기상 조건 조합 특징
            features[i][0] * features[i][3],  # 기온 * 일사량
            features[i][2] * features[i][3],  # 일조 * 일사량
            # 강수량이 발전에 미치는 영향
            1 if features[i][1] < 0.1 else 0,  # 무강수 여부
        ]
        pattern_features.append(pattern_feat)
    
    return np.array(pattern_features)


def xgb_stacking_model(X_train, y_train, X_val, y_val, X_test, y_test, plotting=False):
    """
    LSTM 예측을 포함한 스태킹 XGBoost 모델
    """
    # XGBoost 모델 정의 (하이퍼파라미터 튜닝)
    xgb_regressor = xgb.XGBRegressor(
        gamma=0.5, 
        n_estimators=300, 
        learning_rate=0.08, 
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    print("XGBoost 스태킹 모델 학습 중...")
    # 모델 학습
    xgb_regressor.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    # 테스트 데이터에 대한 예측
    pred_test = xgb_regressor.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    rmse = calculate_rmse(y_test, pred_test)
    mape = calculate_mape(y_test, pred_test)
    
    # 특성 중요도 출력
    feature_names = ['기온', '강수량', '일조', '일사량'] + \
                      ['패턴', '연간_sin', '연간_cos', '일간_sin', '일간_cos', 
                       '기온×일사량', '일조×일사량', '무강수여부'] + \
                      ['LSTM_예측값'] # 스태킹 특성 이름 변경
    
    importance_dict = dict(zip(feature_names, xgb_regressor.feature_importances_))
    print("\n=== XGBoost 스태킹 모델 특성 중요도 ===")
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # 결과 시각화
    if plotting:
        plt.figure(figsize=(15, 10))
        
        # 예측 결과 플롯
        plt.subplot(2, 2, 1)
        plt.plot(y_test[:200], label='Actual', alpha=0.7)
        plt.plot(pred_test[:200], label='Predicted', alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("발전량 (MWh)")
        plt.title(f"XGBoost 스태킹 모델 - MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}%")
        plt.legend()
        
        # 산점도
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('실제값 (Actual)')
        plt.ylabel('예측값 (Predicted)')
        plt.title('실제값 vs 예측값')
        
        # 특성 중요도
        plt.subplot(2, 2, 3)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_regressor.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('중요도 (Importance)')
        plt.title('특성 중요도')
        
        # 잔차 분석
        plt.subplot(2, 2, 4)
        residuals = y_test - pred_test
        plt.scatter(pred_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값 (Predicted)')
        plt.ylabel('잔차 (Residuals)')
        plt.title('잔차 플롯')
        
        plt.tight_layout()
        plt.show()
        
    return mae, rmse, mape, xgb_regressor

def create_sequences_and_split_with_patterns(features, targets, pattern_features, 
                                            seq_len=24, test_size=0.2, val_size=0.1):
    """
    패턴 정보를 포함한 시퀀스 데이터 생성 및 분할
    """
    # 스케일링 (기상 피처와 타겟만 스케일)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    # pattern_features의 첫 열은 '패턴 라벨' (정수)
    # 절대 스케일링하지 않고 정수로 유지
    patterns = pattern_features[:, 0].astype(int)  # e.g. 0,1,2,...,n_patterns-1
    
    # 데이터셋 생성
    dataset = SolarPatternDataset(features_scaled, targets_scaled, patterns, seq_len)
    
    # 데이터 분할 (비율 기반)
    dataset_size = len(dataset)
    test_split = int(dataset_size * test_size)
    val_split = int(dataset_size * val_size)
    train_split = dataset_size - test_split - val_split
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_split, val_split, test_split]
    )
    
    return (train_dataset, val_dataset, test_dataset, 
            feature_scaler, target_scaler)

if __name__ == "__main__":
    try:
        print("데이터 로딩 중...")
        data_df = pd.read_csv(data_path)
        
        # 1. 데이터 전처리
        features, targets = preprocess_data(data_df)
        if features is None or targets is None:
            raise ValueError("데이터 전처리에 실패했습니다. 원본 데이터를 확인해주세요.")

        # 2. 패턴 추출 및 패턴 기반 특성 생성
        print("\n패턴 추출 중...")
        pattern_extractor = PatternExtractor(n_patterns=5)
        pattern_features = create_pattern_features(features, targets, pattern_extractor)
        
        seq_len = 24
        
        # 3. 데이터셋 시간순 분할 (70% train, 15% validation, 15% test)
        print("\n시간순으로 데이터 분할...")
        total_size = len(features)
        train_end = int(total_size * 0.7)
        val_end = int(total_size * 0.85)

        features_train, targets_train, pattern_train = features[:train_end], targets[:train_end], pattern_features[:train_end]
        features_val, targets_val, pattern_val = features[train_end:val_end], targets[train_end:val_end], pattern_features[train_end:val_end]
        features_test, targets_test, pattern_test = features[val_end:], targets[val_end:], pattern_features[val_end:]
        
        print(f"Train set size: {len(features_train)}")
        print(f"Validation set size: {len(features_val)}")
        print(f"Test set size: {len(features_test)}")

        # 4. 데이터 스케일링 및 PyTorch Dataset/DataLoader 생성
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        features_train_scaled = feature_scaler.fit_transform(features_train)
        features_val_scaled = feature_scaler.transform(features_val)
        features_test_scaled = feature_scaler.transform(features_test)

        targets_train_scaled = target_scaler.fit_transform(targets_train.reshape(-1, 1)).flatten()
        targets_val_scaled = target_scaler.transform(targets_val.reshape(-1, 1)).flatten()
        targets_test_scaled = target_scaler.transform(targets_test.reshape(-1, 1)).flatten()
        
        train_dataset = SolarPatternDataset(features_train_scaled, targets_train_scaled, pattern_train[:, 0], seq_len)
        val_dataset = SolarPatternDataset(features_val_scaled, targets_val_scaled, pattern_val[:, 0], seq_len)
        test_dataset = SolarPatternDataset(features_test_scaled, targets_test_scaled, pattern_test[:, 0], seq_len)
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 5. LSTM 모델 생성 및 학습
        model = LSTM_Pattern(weather_feature_dim=4, pattern_embedding_dim=8, 
                               hidden_dim=128, num_layers=2, n_patterns=5)
        # CUDA로 이동
        model.to(device)
        print("\n" + "="*60)
        print("LSTM 모델 학습 시작...")
        print("="*60)
        train_losses, val_losses = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,
            lr=0.001
        )
        
        # 6. LSTM 모델 성능 평가
        print("\nLSTM 테스트 예측 및 평가 중...")
        predictions_scaled, actuals_scaled = model.predict(test_loader)
        predictions_original = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        actuals_original = target_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()
        
        mae_lstm = mean_absolute_error(actuals_original, predictions_original)
        rmse_lstm = calculate_rmse(actuals_original, predictions_original)
        mape_lstm = calculate_mape(actuals_original, predictions_original)
        
        print(f"\n=== LSTM 모델 성능 평가 ===")
        print(f"MAE: {mae_lstm:.4f}")
        print(f"RMSE: {rmse_lstm:.4f}")
        print(f"MAPE: {mape_lstm:.4f}%")
        
        # 7. 스태킹을 위한 예측값 생성 (XGBoost의 새로운 특성)
        print("\n" + "="*60)
        print("스태킹을 위한 LSTM 예측값 생성 중...")
        print("="*60)
        # 순서 유지를 위해 shuffle=False 로 DataLoader 재생성
        train_loader_stack = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader_stack = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        lstm_preds_train_scaled, _ = model.predict(train_loader_stack)
        lstm_preds_val_scaled, _ = model.predict(val_loader_stack)
        # 테스트 예측값은 이미 계산됨 (predictions_scaled)
        
        lstm_preds_train = target_scaler.inverse_transform(lstm_preds_train_scaled.reshape(-1, 1))
        lstm_preds_val = target_scaler.inverse_transform(lstm_preds_val_scaled.reshape(-1, 1))
        lstm_preds_test = predictions_original.reshape(-1, 1) # 이미 원본 스케일

        # 8. XGBoost 학습을 위한 데이터 준비
        # LSTM 예측값과 원래 특성들을 결합
        # 시퀀스 생성으로 인해 데이터 길이가 줄어든 것을 반영 (앞부분 seq_len 만큼 제거)
        X_train_xgb = np.hstack([features_train[seq_len:], pattern_train[seq_len:], lstm_preds_train])
        y_train_xgb = targets_train[seq_len:]
        
        X_val_xgb = np.hstack([features_val[seq_len:], pattern_val[seq_len:], lstm_preds_val])
        y_val_xgb = targets_val[seq_len:]
        
        X_test_xgb = np.hstack([features_test[seq_len:], pattern_test[seq_len:], lstm_preds_test])
        y_test_xgb = targets_test[seq_len:]

        # 9. XGBoost 스태킹 모델 학습 및 평가
        print("\n" + "="*60)
        print("XGBoost 스태킹 앙상블 모델 학습 시작...")
        print("="*60)
        mae_xgb, rmse_xgb, mape_xgb, xgb_model = xgb_stacking_model(
            X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb, X_test_xgb, y_test_xgb, plotting=True
        )
        
        print(f"\n=== XGBoost 스태킹 모델 성능 평가 ===")
        print(f"MAE: {mae_xgb:.4f}")
        print(f"RMSE: {rmse_xgb:.4f}")
        print(f"MAPE: {mape_xgb:.4f}%")
        
        # 최종 비교
        print(f"\n{'='*60}")
        print("최종 모델 성능 비교")
        print(f"{'='*60}")
        print(f"LSTM             : MAE={mae_lstm:.4f}, RMSE={rmse_lstm:.4f}, MAPE={mape_lstm:.2f}%")
        print(f"XGBoost (Stacked): MAE={mae_xgb:.4f}, RMSE={rmse_xgb:.4f}, MAPE={mape_xgb:.2f}%")
        
        # 학습 곡선 시각화
        if train_losses and val_losses:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('LSTM Training History')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(actuals_original[:200], label='Actual', alpha=0.7)
            plt.plot(predictions_original[:200], label='LSTM Predicted', alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('태양광 발전량 (MWh)')
            plt.title('LSTM Prediction Results')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()