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
        # --- FIX START ---
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


class GRU_Pattern(nn.Module):
    """
    패턴 정보를 활용한 GRU 모델
    """
    def __init__(self, weather_feature_dim=4, pattern_embedding_dim=8, hidden_dim=64, num_layers=2, n_patterns=5):
        super(GRU_Pattern, self).__init__()
        
        self.n_patterns = n_patterns
        
        # Pattern Embedding Layer
        self.pattern_embed = nn.Embedding(n_patterns, pattern_embedding_dim)
        
        # GRU (기상 데이터 + 패턴 임베딩)
        gru_input_size = weather_feature_dim + pattern_embedding_dim
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_dim, 
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
        
        # GRU
        gru_out, h_n = self.gru(combined_features)
        
        # Self-attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Use last hidden state
        out = attn_out[:, -1, :]  # (batch, hidden_dim)
        
        # FC layers with batch normalization
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
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
        
        # 순서 유지를 위한 shuffle=False DataLoader
        train_loader_stack = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader_stack = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 5. LSTM 모델 생성 및 학습
        print("\n" + "="*60)
        print("LSTM 모델 학습 시작...")
        print("="*60)
        
        lstm_model = LSTM_Pattern(weather_feature_dim=4, pattern_embedding_dim=8, 
                                 hidden_dim=128, num_layers=2, n_patterns=5)
        
        lstm_train_losses, lstm_val_losses = lstm_model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,
            lr=0.001
        )
        
        # 6. GRU 모델 생성 및 학습
        print("\n" + "="*60)
        print("GRU 모델 학습 시작...")
        print("="*60)
        
        gru_model = GRU_Pattern(weather_feature_dim=4, pattern_embedding_dim=8, 
                               hidden_dim=128, num_layers=2, n_patterns=5)
        
        gru_train_losses, gru_val_losses = gru_model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,
            lr=0.001
        )
        
        # 7. 개별 모델 성능 평가
        print("\n" + "="*60)
        print("개별 모델 성능 평가...")
        print("="*60)
        
        # LSTM 예측 및 평가
        lstm_predictions_scaled, lstm_actuals_scaled = lstm_model.predict(test_loader)
        lstm_predictions_original = target_scaler.inverse_transform(lstm_predictions_scaled.reshape(-1, 1)).flatten()
        lstm_actuals_original = target_scaler.inverse_transform(lstm_actuals_scaled.reshape(-1, 1)).flatten()
        
        mae_lstm = mean_absolute_error(lstm_actuals_original, lstm_predictions_original)
        rmse_lstm = calculate_rmse(lstm_actuals_original, lstm_predictions_original)
        mape_lstm = calculate_mape(lstm_actuals_original, lstm_predictions_original)
        
        print(f"\n=== LSTM 모델 성능 평가 ===")
        print(f"MAE: {mae_lstm:.4f}")
        print(f"RMSE: {rmse_lstm:.4f}")
        print(f"MAPE: {mape_lstm:.4f}%")
        
        # GRU 예측 및 평가
        gru_predictions_scaled, gru_actuals_scaled = gru_model.predict(test_loader)
        gru_predictions_original = target_scaler.inverse_transform(gru_predictions_scaled.reshape(-1, 1)).flatten()
        gru_actuals_original = target_scaler.inverse_transform(gru_actuals_scaled.reshape(-1, 1)).flatten()
        
        mae_gru = mean_absolute_error(gru_actuals_original, gru_predictions_original)
        rmse_gru = calculate_rmse(gru_actuals_original, gru_predictions_original)
        mape_gru = calculate_mape(gru_actuals_original, gru_predictions_original)
        
        print(f"\n=== GRU 모델 성능 평가 ===")
        print(f"MAE: {mae_gru:.4f}")
        print(f"RMSE: {rmse_gru:.4f}")
        print(f"MAPE: {mape_gru:.4f}%")
        
        # 8. 스태킹을 위한 예측값 생성
        print("\n" + "="*60)
        print("스태킹을 위한 LSTM & GRU 예측값 생성 중...")
        print("="*60)
        
        # Train set 예측값 생성
        lstm_preds_train_scaled, _ = lstm_model.predict(train_loader_stack)
        gru_preds_train_scaled, _ = gru_model.predict(train_loader_stack)
        
        # Validation set 예측값 생성
        lstm_preds_val_scaled, _ = lstm_model.predict(val_loader_stack)
        gru_preds_val_scaled, _ = gru_model.predict(val_loader_stack)
        
        # 원본 스케일로 변환
        lstm_preds_train = target_scaler.inverse_transform(lstm_preds_train_scaled.reshape(-1, 1))
        gru_preds_train = target_scaler.inverse_transform(gru_preds_train_scaled.reshape(-1, 1))
        
        lstm_preds_val = target_scaler.inverse_transform(lstm_preds_val_scaled.reshape(-1, 1))
        gru_preds_val = target_scaler.inverse_transform(gru_preds_val_scaled.reshape(-1, 1))
        
        lstm_preds_test = lstm_predictions_original.reshape(-1, 1)
        gru_preds_test = gru_predictions_original.reshape(-1, 1)
        
        # 9. XGBoost 스태킹 학습을 위한 데이터 준비
        print("\nXGBoost 스태킹을 위한 특성 결합 중...")
        
        # 시퀀스 생성으로 인해 데이터 길이가 줄어든 것을 반영
        X_train_stack = np.hstack([
            features_train[seq_len:],           # 원본 기상 특성 (4개)
            pattern_train[seq_len:],            # 패턴 특성 (8개)
            lstm_preds_train,                   # LSTM 예측값 (1개)
            gru_preds_train                     # GRU 예측값 (1개)
        ])
        y_train_stack = targets_train[seq_len:]
        
        X_val_stack = np.hstack([
            features_val[seq_len:],
            pattern_val[seq_len:],
            lstm_preds_val,
            gru_preds_val
        ])
        y_val_stack = targets_val[seq_len:]
        
        X_test_stack = np.hstack([
            features_test[seq_len:],
            pattern_test[seq_len:],
            lstm_preds_test,
            gru_preds_test
        ])
        y_test_stack = targets_test[seq_len:]
        
        print(f"스태킹 특성 차원: {X_train_stack.shape[1]} (기상 4 + 패턴 8 + LSTM 1 + GRU 1)")
        
        # 10. XGBoost 스태킹 모델 학습 및 평가
        print("\n" + "="*60)
        print("XGBoost 스태킹 앙상블 모델 학습 시작...")
        print("="*60)
        
        # 특성 이름 정의
        feature_names = ['기온', '강수량', '일조', '일사량'] + \
                       ['패턴', '연간_sin', '연간_cos', '일간_sin', '일간_cos', 
                        '기온×일사량', '일조×일사량', '무강수여부'] + \
                       ['LSTM_예측값', 'GRU_예측값']
        
        # XGBoost 모델 정의 및 학습
        xgb_stacking_regressor = xgb.XGBRegressor(
            gamma=0.3,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            reg_alpha=0.05,
            reg_lambda=0.1,
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
        
        mae_stacked = mean_absolute_error(y_test_stack, stacked_pred_test)
        rmse_stacked = calculate_rmse(y_test_stack, stacked_pred_test)
        mape_stacked = calculate_mape(y_test_stack, stacked_pred_test)
        
        print(f"\n=== XGBoost 스태킹 모델 성능 평가 ===")
        print(f"MAE: {mae_stacked:.4f}")
        print(f"RMSE: {rmse_stacked:.4f}")
        print(f"MAPE: {mape_stacked:.4f}%")
        
        # 11. 특성 중요도 분석
        print("\n=== XGBoost 스태킹 모델 특성 중요도 ===")
        importance_dict = dict(zip(feature_names, xgb_stacking_regressor.feature_importances_))
        for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # 12. 최종 성능 비교
        print(f"\n{'='*80}")
        print("최종 모델 성능 비교")
        print(f"{'='*80}")
        print(f"{'모델':<20} {'MAE':<10} {'RMSE':<10} {'MAPE (%)':<10}")
        print("-" * 50)
        print(f"{'LSTM':<20} {mae_lstm:<10.4f} {rmse_lstm:<10.4f} {mape_lstm:<10.2f}")
        print(f"{'GRU':<20} {mae_gru:<10.4f} {rmse_gru:<10.4f} {mape_gru:<10.2f}")
        print(f"{'XGBoost Stacking':<20} {mae_stacked:<10.4f} {rmse_stacked:<10.4f} {mape_stacked:<10.2f}")
        
        # 성능 개선율 계산
        best_individual = min(mae_lstm, mae_gru)
        improvement = ((best_individual - mae_stacked) / best_individual) * 100
        print(f"\n스태킹으로 인한 성능 개선: {improvement:.2f}%")
        
        # 13. 결과 시각화
        print("\n결과 시각화 중...")
        plt.figure(figsize=(20, 15))
        
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
        plt.title(f'LSTM 예측 결과\nMAE: {mae_lstm:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 4)
        plt.plot(gru_actuals_original[:test_range], label='Actual', alpha=0.8, linewidth=2)
        plt.plot(gru_predictions_original[:test_range], label='GRU', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('발전량 (MWh)')
        plt.title(f'GRU 예측 결과\nMAE: {mae_gru:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 5)
        plt.plot(y_test_stack[:test_range], label='Actual', alpha=0.8, linewidth=2)
        plt.plot(stacked_pred_test[:test_range], label='Stacked', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('발전량 (MWh)')
        plt.title(f'XGBoost 스태킹 결과\nMAE: {mae_stacked:.3f}')
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
        plt.show()
        
        # 14. 성능 메트릭 비교 막대 그래프
        plt.figure(figsize=(15, 5))
        
        models = ['LSTM', 'GRU', 'XGBoost\nStacking']
        mae_scores = [mae_lstm, mae_gru, mae_stacked]
        rmse_scores = [rmse_lstm, rmse_gru, rmse_stacked]
        mape_scores = [mape_lstm, mape_gru, mape_stacked]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.subplot(1, 3, 1)
        plt.bar(x, mae_scores, width, label='MAE', alpha=0.8)
        plt.xlabel('모델')
        plt.ylabel('MAE')
        plt.title('MAE 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.bar(x, rmse_scores, width, label='RMSE', alpha=0.8, color='orange')
        plt.xlabel('모델')
        plt.ylabel('RMSE')
        plt.title('RMSE 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.bar(x, mape_scores, width, label='MAPE (%)', alpha=0.8, color='green')
        plt.xlabel('모델')
        plt.ylabel('MAPE (%)')
        plt.title('MAPE 비교')
        plt.xticks(x, models)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*80}")
        print("모델 학습 및 평가 완료!")
        print(f"{'='*80}")
        
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()