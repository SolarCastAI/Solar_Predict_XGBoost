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
from sklearn.linear_model import Ridge, RidgeCV  # 릿지 회귀 추가
from sklearn.pipeline import Pipeline
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


def ridge_regression_model(X_train, y_train, X_val, y_val, X_test, y_test, plotting=False):
    """
    릿지 회귀 모델 학습 및 평가
    """
    print("릿지 회귀 모델 학습 중...")
    
    # 교차 검증을 통한 최적 alpha 값 찾기
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
    
    # 데이터 표준화를 포함한 파이프라인 생성
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', ridge_cv)
    ])
    
    # 모델 학습
    ridge_pipeline.fit(X_train, y_train)
    
    # 최적 alpha 값 출력
    best_alpha = ridge_pipeline.named_steps['ridge'].alpha_
    print(f"최적 alpha 값: {best_alpha}")
    
    # 예측
    pred_val = ridge_pipeline.predict(X_val)
    pred_test = ridge_pipeline.predict(X_test)
    
    # 성능 평가
    mae_val = mean_absolute_error(y_val, pred_val)
    rmse_val = calculate_rmse(y_val, pred_val)
    mape_val = calculate_mape(y_val, pred_val)
    
    mae_test = mean_absolute_error(y_test, pred_test)
    rmse_test = calculate_rmse(y_test, pred_test)
    mape_test = calculate_mape(y_test, pred_test)
    
    print(f"릿지 회귀 검증 성능 - MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.4f}%")
    print(f"릿지 회귀 테스트 성능 - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, MAPE: {mape_test:.4f}%")
    
    # 계수 분석
    ridge_model = ridge_pipeline.named_steps['ridge']
    feature_names = ['기온', '강수량', '일조', '일사량'] + \
                   ['패턴', '연간_sin', '연간_cos', '일간_sin', '일간_cos', 
                    '기온×일사량', '일조×일사량', '무강수여부']
    
    print("\n=== 릿지 회귀 계수 분석 ===")
    coefficients = ridge_model.coef_
    coef_dict = dict(zip(feature_names, coefficients))
    
    # 계수의 절댓값 기준으로 정렬하여 중요도 표시
    for feature, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {coef:.4f}")
    
    print(f"절편 (Intercept): {ridge_model.intercept_:.4f}")
    
    # 결과 시각화
    if plotting:
        plt.figure(figsize=(15, 10))
        
        # 예측 결과 플롯
        plt.subplot(2, 3, 1)
        test_range = min(200, len(y_test))
        plt.plot(y_test[:test_range], label='실제값', alpha=0.8, linewidth=2)
        plt.plot(pred_test[:test_range], label='예측값', alpha=0.7)
        plt.xlabel("시간")
        plt.ylabel("발전량 (MWh)")
        plt.title(f"릿지 회귀 예측 결과\nMAE: {mae_test:.3f}, RMSE: {rmse_test:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 산점도
        plt.subplot(2, 3, 2)
        plt.scatter(y_test, pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('실제값')
        plt.ylabel('예측값')
        plt.title('실제값 vs 예측값')
        plt.grid(True, alpha=0.3)
        
        # 계수 중요도
        plt.subplot(2, 3, 3)
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=True)
        
        colors = ['red' if x < 0 else 'blue' for x in coef_df['coefficient']]
        plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.7)
        plt.xlabel('계수 값')
        plt.title('릿지 회귀 계수')
        plt.grid(True, alpha=0.3)
        
        # 잔차 분석
        plt.subplot(2, 3, 4)
        residuals = y_test - pred_test
        plt.scatter(pred_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값')
        plt.ylabel('잔차')
        plt.title('잔차 플롯')
        plt.grid(True, alpha=0.3)
        
        # 잔차 히스토그램
        plt.subplot(2, 3, 5)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.title('잔차 분포')
        plt.grid(True, alpha=0.3)
        
        # Alpha 값별 성능
        plt.subplot(2, 3, 6)
        # CV 점수 시각화
        cv_scores = []
        for alpha in alphas:
            ridge_temp = Ridge(alpha=alpha)
            ridge_temp_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', ridge_temp)
            ])
            ridge_temp_pipeline.fit(X_train, y_train)
            val_pred = ridge_temp_pipeline.predict(X_val)
            cv_scores.append(mean_absolute_error(y_val, val_pred))
        
        plt.semilogx(alphas, cv_scores, 'bo-')
        plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'최적 α={best_alpha}')
        plt.xlabel('Alpha (정규화 강도)')
        plt.ylabel('검증 MAE')
        plt.title('Alpha 값에 따른 성능')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return mae_test, rmse_test, mape_test, ridge_pipeline