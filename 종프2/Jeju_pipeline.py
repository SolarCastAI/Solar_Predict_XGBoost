import torch
import torch.nn as nn
import torch.optim as optim
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

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
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

        self.X = np.array(self.X, dtype=np.float32)        # (N, seq_len, feature_dim)
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


class CNN_LSTM_Pattern(nn.Module):
    """
    패턴 정보를 활용한 CNN + LSTM 모델
    """
    def __init__(self, input_dim=5, seq_len=24, hidden_dim=64, num_layers=2, n_patterns=5):
        super(CNN_LSTM_Pattern, self).__init__()
        
        self.n_patterns = n_patterns
        
        # Pattern Embedding Layer
        self.pattern_embed = nn.Embedding(n_patterns, 8)
        
        # CNN layers (기상 데이터용)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
        # LSTM (CNN 출력 + 패턴 임베딩)
        self.lstm = nn.LSTM(input_size=64+8, hidden_size=hidden_dim, 
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
        
        # CNN for weather data
        weather_data = weather_data.permute(0, 2, 1)  # (batch, features, seq_len)
        conv_out = self.conv1(weather_data)
        conv_out = self.relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.pool(conv_out)  # (batch, 64, seq_len//2)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq_len//2, 64)
        
        # Pattern embedding
        pattern_emb = self.pattern_embed(pattern_data)  # (batch, seq_len, 8)
        
        # Adjust sequence length for pattern embedding
        target_seq_len = conv_out.size(1)
        if pattern_emb.size(1) != target_seq_len:
            # Simple interpolation or pooling to match sequence lengths
            pattern_emb = pattern_emb[:, :target_seq_len, :]
        
        # Combine CNN output with pattern embedding
        combined_features = torch.cat([conv_out, pattern_emb], dim=2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state
        out = attn_out[:, -1, :]  # (batch, hidden_dim)
        
        # FC layers with batch normalization
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
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
                predictions.extend(preds.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)


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


def xgb_model_with_patterns(X_train, y_train, X_val, y_val, pattern_features_train, 
                            pattern_features_val, plotting=False):
    """
    패턴 정보를 활용한 XGBoost 모델
    """
    # 패턴 특징과 원본 특징 결합
    X_train_combined = np.column_stack([X_train, pattern_features_train])
    X_val_combined = np.column_stack([X_val, pattern_features_val])
    
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
    
    # 모델 학습
    xgb_regressor.fit(
        X_train_combined, 
        y_train, 
        eval_set=[(X_val_combined, y_val)], 
        verbose=False
    )
    
    # 검증 데이터에 대한 예측
    pred_val = xgb_regressor.predict(X_val_combined)
    mae = mean_absolute_error(y_val, pred_val)
    rmse = calculate_rmse(y_val, pred_val)
    # <<< 수정된 부분 시작 >>>
    mape = calculate_mape(y_val, pred_val)
    # <<< 수정된 부분 끝 >>>
    
    # 특성 중요도 출력
    feature_names = ['기온', '강수량', '일조', '일사량'] + \
                    ['패턴', '연간_sin', '연간_cos', '일간_sin', '일간_cos', 
                     '기온×일사량', '일조×일사량', '무강수여부']
    
    importance_dict = dict(zip(feature_names, xgb_regressor.feature_importances_))
    print("\n=== XGBoost 특성 중요도 ===")
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # 결과 시각화
    if plotting:
        plt.figure(figsize=(15, 10))
        
        # 예측 결과 플롯
        plt.subplot(2, 2, 1)
        plt.plot(y_val[:200], label='Actual', alpha=0.7)
        plt.plot(pred_val[:200], label='Predicted', alpha=0.7)
        plt.xlabel("Time")
        plt.ylabel("DC Power")
        plt.title(f"XGBoost with Patterns - MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}%")
        plt.legend()
        
        # 산점도
        plt.subplot(2, 2, 2)
        plt.scatter(y_val, pred_val, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        
        # 특성 중요도
        plt.subplot(2, 2, 3)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_regressor.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        
        # 잔차 분석
        plt.subplot(2, 2, 4)
        residuals = y_val - pred_val
        plt.scatter(pred_val, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
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
        
        # 1. 정의된 전처리 함수를 호출하여 결측값을 먼저 처리합니다.
        features, targets = preprocess_data(data_df)
        
        # 전처리가 실패했는지 확인 (예: 필요한 컬럼이 없는 경우)
        if features is None or targets is None:
            raise ValueError("데이터 전처리에 실패했습니다. 원본 데이터를 확인해주세요.")

        print(f"데이터 크기: {len(features)} 행")
        print("패턴 추출 중...")
        
        # 2. 깨끗해진 데이터를 기반으로 K-means를 활용하여 패턴을 추출합니다.
        pattern_extractor = PatternExtractor(n_patterns=5)
        pattern_features = create_pattern_features(features, targets, pattern_extractor)
        
        print("데이터 전처리 중...")
        seq_len = 24
        
        # 패턴 정보를 포함한 시퀀스 데이터 생성
        (train_dataset, val_dataset, test_dataset, 
        feature_scaler, target_scaler) = create_sequences_and_split_with_patterns(
            features, targets, pattern_features, seq_len=seq_len
        )
        
        # DataLoader 생성
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # CNN + LSTM 모델 생성 및 학습
        model = CNN_LSTM_Pattern(input_dim=5, seq_len=seq_len, 
                                 hidden_dim=128, num_layers=2, n_patterns=5)
        print("CNN+LSTM 모델 생성 완료")
        
        print("\n" + "="*60)
        print("CNN+LSTM 모델 학습 시작...")
        print("="*60)
        train_losses, val_losses = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,  # 에포크를 10으로 수정
            lr=0.001
        )
        
        # CNN+LSTM 테스트 예측
        print("\nCNN+LSTM 테스트 예측 중...")
        predictions, actuals = model.predict(test_loader)

        # 예측값과 실제값에 NaN이 있는지 확인 (디버깅용)
        if np.isnan(predictions).any() or np.isnan(actuals).any():
            print("Warning: 예측 또는 실제값에 여전히 NaN이 포함되어 있습니다.")
            # NaN을 0이나 평균값으로 대체하여 임시로 오류를 피할 수 있습니다.
            predictions = np.nan_to_num(predictions)
            actuals = np.nan_to_num(actuals)
        
        # 원본 스케일로 변환
        predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 성능 평가
        mae_cnn_lstm = mean_absolute_error(actuals_original, predictions_original)
        rmse_cnn_lstm = calculate_rmse(actuals_original, predictions_original)
        mape_cnn_lstm = calculate_mape(actuals_original, predictions_original)
        
        print(f"\n=== CNN+LSTM 모델 성능 평가 ===")
        print(f"MAE: {mae_cnn_lstm:.4f}")
        print(f"RMSE: {rmse_cnn_lstm:.4f}")
        print(f"MAPE: {mape_cnn_lstm:.4f}%")
        
        # XGBoost 모델 학습 (비교용)
        print("\n" + "="*60)
        print("XGBoost 모델 학습 시작...")
        print("="*60)
        
        # 데이터 분할 (XGBoost용)
        split_idx = int(len(features) * 0.8)
        X_train_xgb, X_test_xgb = features[:split_idx], features[split_idx:]
        y_train_xgb, y_test_xgb = targets[:split_idx], targets[split_idx:]
        pattern_train_xgb = pattern_features[:split_idx]
        pattern_test_xgb = pattern_features[split_idx:]
        
        # 검증 데이터 분할
        val_split_idx = int(len(X_train_xgb) * 0.8)
        X_train_final = X_train_xgb[:val_split_idx]
        X_val_final = X_train_xgb[val_split_idx:]
        y_train_final = y_train_xgb[:val_split_idx]
        y_val_final = y_train_xgb[val_split_idx:]
        pattern_train_final = pattern_train_xgb[:val_split_idx]
        pattern_val_final = pattern_train_xgb[val_split_idx:]
        
        # XGBoost 학습
        mae_xgb, rmse_xgb, mape_xgb, xgb_model = xgb_model_with_patterns(
            X_train_final, y_train_final, X_val_final, y_val_final,
            pattern_train_final, pattern_val_final, plotting=True
        )
        
        print(f"\n=== XGBoost 모델 성능 평가 ===")
        print(f"MAE: {mae_xgb:.4f}")
        print(f"RMSE: {rmse_xgb:.4f}")
        print(f"MAPE: {mape_xgb:.4f}%")
        
        # 최종 비교
        print(f"\n{'='*60}")
        print("최종 모델 성능 비교")
        print(f"{'='*60}")
        print(f"CNN+LSTM: MAE={mae_cnn_lstm:.4f}, RMSE={rmse_cnn_lstm:.4f}, MAPE={mape_cnn_lstm:.2f}%")
        print(f"XGBoost:  MAE={mae_xgb:.4f}, RMSE={rmse_xgb:.4f}, MAPE={mape_xgb:.2f}%")
        
        # 학습 곡선 시각화
        if train_losses and val_losses:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('CNN+LSTM Training History')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(actuals_original[:200], label='Actual', alpha=0.7)
            plt.plot(predictions_original[:200], label='CNN+LSTM Predicted', alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Solar Power (MWh)')
            plt.title('CNN+LSTM Prediction Results')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()