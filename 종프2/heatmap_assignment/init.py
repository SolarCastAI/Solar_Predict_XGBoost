"""
계절별 기상 패턴 클러스터링 및 도로 열위험도 예측 (과적합 방지 개선)
- 태양광 발전 데이터를 활용한 계절별 K-means 클러스터링
- LSTM_Pattern / GRU_Pattern 모델 (정규화 강화)
- XGBoost 스태킹 앙상블
- 평가지표: NMAE, R², MAPE, Accuracy
"""

import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 재현성을 위한 시드 설정
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 평가 지표 함수 
def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_nmae(y_true, y_pred):
    """NMAE (Normalized MAE): MAE를 실제값의 범위로 정규화"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if len(y_true_valid) == 0:
        return 0.0
    data_range = np.max(y_true_valid) - np.min(y_true_valid)
    if data_range == 0:
        return 0.0
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    return mae / data_range

def calculate_mape(y_true, y_pred):
    """MAPE (Mean Absolute Percentage Error): 평균 절대 백분율 오차"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if len(y_true_valid) == 0:
        return 0.0
    return np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100

def calculate_accuracy_within_threshold(y_true, y_pred, threshold_percent=10):
    """임계값 내 정확도: 예측값이 실제값의 ±threshold% 이내에 있는 비율"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if len(y_true_valid) == 0:
        return 0.0
    
    relative_error = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100
    within_threshold = (relative_error <= threshold_percent).sum() / len(y_true_valid) * 100
    
    return within_threshold


class SolarDataPreprocessor:
    """태양광 발전 데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_solar_data(self, filepath):
        """태양광 발전 데이터 로드"""
        print("=" * 80)
        print("1. 태양광 발전 데이터 로드")
        print("=" * 80)
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        print(f"\n원본 데이터 shape: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        print(f"\n데이터 미리보기:")
        print(df.head())
        
        return df
    
    def handle_missing_outliers(self, df):
        """결측치 및 이상치 처리"""
        print("\n" + "=" * 80)
        print("2. 결측치 및 이상치 처리")
        print("=" * 80)
        
        initial_rows = len(df)
        
        print("\n결측치 현황:")
        print(df.isnull().sum())
        
        df['발전일자'] = pd.to_datetime(df['발전일자'], errors='coerce')
        df = df.dropna(subset=['발전일자'])
        
        if '기온' in df.columns:
            print(f"\n기온 범위: {df['기온'].min():.1f}°C ~ {df['기온'].max():.1f}°C")
            df.loc[(df['기온'] < -30) | (df['기온'] > 50), '기온'] = np.nan
        
        if '습도' in df.columns:
            valid_humidity = df['습도'].notna()
            if valid_humidity.sum() > 0:
                print(f"습도 범위: {df.loc[valid_humidity, '습도'].min():.1f}% ~ {df.loc[valid_humidity, '습도'].max():.1f}%")
            df.loc[(df['습도'] < 0) | (df['습도'] > 100), '습도'] = np.nan
        
        if '강우량(mm)' in df.columns:
            df.loc[df['강우량(mm)'] < 0, '강우량(mm)'] = 0
        
        if '적설량(mm)' in df.columns:
            df.loc[df['적설량(mm)'] < 0, '적설량(mm)'] = 0
        
        if '풍속' in df.columns:
            df.loc[df['풍속'] < 0, '풍속'] = 0
        
        numeric_cols = ['기온', '강우량(mm)', '습도', '적설량(mm)', '풍속', '적운량(10분위)', '적운량(3분위)']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=3)
        
        df = df.dropna(subset=['기온'])
        
        print(f"\n처리 완료: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} rows 제거)")
        
        return df
    
    def add_temporal_features(self, df):
        """시간 관련 파생변수 생성"""
        print("\n" + "=" * 80)
        print("3. 시간 파생변수 생성")
        print("=" * 80)
        
        df['시간'] = df['발전일자'].dt.hour
        df['요일'] = df['발전일자'].dt.dayofweek
        df['월'] = df['발전일자'].dt.month
        df['일'] = df['발전일자'].dt.day
        
        df['계절'] = df['월'].apply(lambda x: (x % 12 + 3) // 3)
        df['계절명'] = df['계절'].map({1: '봄', 2: '여름', 3: '가을', 4: '겨울'})
        
        df['시간_sin'] = np.sin(2 * np.pi * df['시간'] / 24)
        df['시간_cos'] = np.cos(2 * np.pi * df['시간'] / 24)
        df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
        df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
        
        print(f"\n생성된 시간 변수: 시간, 요일, 월, 계절, 계절명, 시간_sin, 시간_cos, 월_sin, 월_cos")
        print(f"\n계절별 데이터 분포:")
        print(df['계절명'].value_counts().sort_index())
        
        return df


class SeasonalWeatherPatternClustering:
    """계절별 K-means 클러스터링을 통한 기상 패턴 분석"""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.seasonal_kmeans = {}
        self.seasonal_scalers = {}
        
    def fit_predict(self, df):
        """계절별 기상 패턴 클러스터링"""
        print("\n" + "=" * 80)
        print("4. 계절별 K-means 클러스터링 - 기상 패턴 분석")
        print("=" * 80)
        
        cluster_features = ['기온', '강우량(mm)', '습도', '적설량(mm)', '풍속', '적운량(10분위)']
        
        for col in cluster_features:
            if col in df.columns:
                df[f'{col}_filled'] = df[col].fillna(df[col].mean())
            else:
                df[f'{col}_filled'] = 0
        
        cluster_features_filled = [f'{col}_filled' for col in cluster_features]
        
        df['기상패턴'] = -1
        
        for season_name in ['봄', '여름', '가을', '겨울']:
            season_mask = df['계절명'] == season_name
            season_data = df.loc[season_mask, cluster_features_filled]
            
            if len(season_data) < self.n_clusters:
                print(f"\n{season_name}: 데이터가 부족하여 클러스터링 건너뜀")
                continue
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(season_data)
            
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            season_num = {'봄': 1, '여름': 2, '가을': 3, '겨울': 4}[season_name]
            unique_labels = season_num * 10 + cluster_labels
            
            df.loc[season_mask, '기상패턴'] = unique_labels
            
            self.seasonal_kmeans[season_name] = kmeans
            self.seasonal_scalers[season_name] = scaler
            
            print(f"\n{season_name} 계절 클러스터 (총 {len(season_data)}개 데이터):")
            print("-" * 80)
            
            for i in range(self.n_clusters):
                cluster_mask = df['기상패턴'] == unique_labels[cluster_labels == i][0] if (cluster_labels == i).sum() > 0 else df['기상패턴'] == -1
                count = cluster_mask.sum()
                if count > 0:
                    print(f"\n  클러스터 {season_num}{i} (N={count}):")
                    print(f"    기온: {df.loc[cluster_mask, '기온_filled'].mean():.1f}°C")
                    print(f"    강우량: {df.loc[cluster_mask, '강우량(mm)_filled'].mean():.2f}mm")
                    print(f"    습도: {df.loc[cluster_mask, '습도_filled'].mean():.1f}%")
        
        df.loc[df['기상패턴'] == -1, '기상패턴'] = 0
        
        print(f"\n전체 기상패턴 분포:")
        print(df['기상패턴'].value_counts().sort_index())
        
        return df


class RoadDataPreprocessor:
    """도로 열위험도 데이터 전처리 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        
    def load_and_parse_csv(self, filepath):
        """CSV 파일 로드 및 파싱"""
        print("\n" + "=" * 80)
        print("5. 도로 데이터 로드 및 파싱")
        print("=" * 80)
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        
        print(f"\n원본 데이터 shape: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        
        data_frames = []
        
        if len(df.columns) >= 4:
            weather_df = df.iloc[:, 0:4].copy()
            weather_df.columns = ['일시', '기온', '습도', '지점명']
            weather_df['데이터소스'] = '기상청'
            data_frames.append(weather_df)
            print(f"\n기상청 데이터: {len(weather_df)} rows")
        
        if len(df.columns) >= 8:
            iot_df = df.iloc[:, 4:8].copy()
            iot_df.columns = ['일시', '기온', '습도', '지점명']
            iot_df['데이터소스'] = 'IoT센서'
            data_frames.append(iot_df)
            print(f"IoT 센서 데이터: {len(iot_df)} rows")
        
        if len(df.columns) >= 11:
            logger1_df = df.iloc[:, 8:11].copy()
            logger1_df.columns = ['일시', '기온', '지점명']
            logger1_df['습도'] = np.nan
            logger1_df['데이터소스'] = '데이터로거_1367'
            data_frames.append(logger1_df)
            print(f"데이터로거 1367: {len(logger1_df)} rows")
        
        if len(df.columns) >= 14:
            logger2_df = df.iloc[:, 11:14].copy()
            logger2_df.columns = ['일시', '기온', '지점명']
            logger2_df['습도'] = np.nan
            logger2_df['데이터소스'] = '데이터로거_1368'
            data_frames.append(logger2_df)
            print(f"데이터로거 1368: {len(logger2_df)} rows")
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        combined_df['일시'] = pd.to_datetime(combined_df['일시'], errors='coerce')
        combined_df['기온'] = pd.to_numeric(combined_df['기온'], errors='coerce')
        combined_df['습도'] = pd.to_numeric(combined_df['습도'], errors='coerce')
        
        print(f"\n통합 데이터 shape: {combined_df.shape}")
        print(f"\n데이터 소스별 분포:")
        print(combined_df['데이터소스'].value_counts())
        
        return combined_df
    
    def handle_missing_outliers(self, df):
        """결측치 및 이상치 처리"""
        print("\n" + "=" * 80)
        print("6. 결측치 및 이상치 처리")
        print("=" * 80)
        
        initial_rows = len(df)
        
        print("\n결측치 현황:")
        print(df.isnull().sum())
        
        df = df.dropna(subset=['일시'])
        
        print(f"\n기온 범위: {df['기온'].min():.1f}°C ~ {df['기온'].max():.1f}°C")
        df.loc[(df['기온'] < -20) | (df['기온'] > 50), '기온'] = np.nan
        
        if '습도' in df.columns:
            valid_humidity = df['습도'].notna()
            if valid_humidity.sum() > 0:
                print(f"습도 범위: {df.loc[valid_humidity, '습도'].min():.1f}% ~ {df.loc[valid_humidity, '습도'].max():.1f}%")
            df.loc[(df['습도'] < 0) | (df['습도'] > 100), '습도'] = np.nan
        
        for location in df['지점명'].unique():
            mask = df['지점명'] == location
            df.loc[mask, '기온'] = df.loc[mask, '기온'].interpolate(method='linear', limit=3)
            if '습도' in df.columns:
                df.loc[mask, '습도'] = df.loc[mask, '습도'].interpolate(method='linear', limit=3)
        
        df = df.dropna(subset=['기온'])
        
        print(f"\n처리 완료: {initial_rows} -> {len(df)} rows ({initial_rows - len(df)} rows 제거)")
        
        return df
    
    def add_temporal_features(self, df):
        """시간 관련 파생변수 생성"""
        print("\n" + "=" * 80)
        print("7. 시간 파생변수 생성")
        print("=" * 80)
        
        df['시간'] = df['일시'].dt.hour
        df['요일'] = df['일시'].dt.dayofweek
        df['월'] = df['일시'].dt.month
        df['계절'] = df['월'].apply(lambda x: (x % 12 + 3) // 3)
        df['계절명'] = df['계절'].map({1: '봄', 2: '여름', 3: '가을', 4: '겨울'})
        
        df['시간_sin'] = np.sin(2 * np.pi * df['시간'] / 24)
        df['시간_cos'] = np.cos(2 * np.pi * df['시간'] / 24)
        df['월_sin'] = np.sin(2 * np.pi * df['월'] / 12)
        df['월_cos'] = np.cos(2 * np.pi * df['월'] / 12)
        
        print(f"\n생성된 시간 변수: 시간, 요일, 월, 계절, 계절명, 시간_sin, 시간_cos, 월_sin, 월_cos")
        
        return df
    
    def merge_with_weather_patterns(self, road_df, solar_df, seasonal_clustering):
        """도로 데이터에 기상 패턴 매핑"""
        print("\n" + "=" * 80)
        print("8. 도로 데이터에 기상 패턴 매핑")
        print("=" * 80)
        
        road_df['습도_filled'] = road_df['습도'].fillna(road_df['습도'].mean())
        road_df['기상패턴'] = 0
        
        for season_name in ['봄', '여름', '가을', '겨울']:
            season_mask = road_df['계절명'] == season_name
            
            if season_mask.sum() == 0:
                print(f"{season_name}: 데이터 없음 (건너뜀)")
                continue
            
            if season_name not in seasonal_clustering.seasonal_kmeans:
                print(f"{season_name}: 클러스터링 모델 없음 (건너뜀)")
                continue
            
            kmeans = seasonal_clustering.seasonal_kmeans[season_name]
            scaler = seasonal_clustering.seasonal_scalers[season_name]
            
            season_road_data = road_df.loc[season_mask].copy()
            
            n_samples = len(season_road_data)
            if n_samples == 0:
                print(f"{season_name}: 필터링 후 데이터 없음 (건너뜀)")
                continue
            
            features_for_clustering = np.zeros((n_samples, 6))
            features_for_clustering[:, 0] = season_road_data['기온'].values
            features_for_clustering[:, 1] = 0
            features_for_clustering[:, 2] = season_road_data['습도_filled'].values
            features_for_clustering[:, 3] = 0
            features_for_clustering[:, 4] = 0
            features_for_clustering[:, 5] = 0
            
            if np.isnan(features_for_clustering).any():
                print(f"{season_name}: NaN 값 발견, 0으로 대체")
                features_for_clustering = np.nan_to_num(features_for_clustering, nan=0.0)
            
            try:
                X_scaled = scaler.transform(features_for_clustering)
                cluster_labels = kmeans.predict(X_scaled)
                
                season_num = {'봄': 1, '여름': 2, '가을': 3, '겨울': 4}[season_name]
                unique_labels = season_num * 10 + cluster_labels
                
                road_df.loc[season_mask, '기상패턴'] = unique_labels
                
                print(f"{season_name}: {season_mask.sum()}개 데이터에 패턴 매핑 완료")
                
            except Exception as e:
                print(f"{season_name}: 패턴 매핑 중 오류 발생 - {str(e)}")
                continue
        
        print(f"\n도로 데이터 기상패턴 분포:")
        pattern_counts = road_df['기상패턴'].value_counts().sort_index()
        if len(pattern_counts) > 0:
            print(pattern_counts)
        else:
            print("매핑된 패턴이 없습니다.")
        
        unassigned = (road_df['기상패턴'] == 0).sum()
        if unassigned > 0:
            print(f"\n경고: {unassigned}개 데이터가 기상패턴이 할당되지 않았습니다.")
        
        return road_df


class LSTM_Pattern(nn.Module):
    """
    과적합 방지가 강화된 LSTM 모델
    - Dropout 비율 증가
    - Layer Normalization 추가
    - Weight Decay 적용
    """
    def __init__(self, weather_feature_dim=4, pattern_embedding_dim=16, hidden_dim=64, num_layers=2, n_patterns=50):
        super(LSTM_Pattern, self).__init__()
        
        self.n_patterns = n_patterns
        self.hidden_dim = hidden_dim
        
        # Pattern Embedding with Dropout
        self.pattern_embed = nn.Embedding(n_patterns, pattern_embedding_dim)
        self.embedding_dropout = nn.Dropout(0.3)
        
        # LSTM with increased dropout
        lstm_input_size = weather_feature_dim + pattern_embedding_dim
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, dropout=0.4)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, 
                                              batch_first=True, dropout=0.3)
        
        # FC layers with stronger regularization
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(0.5)  # 증가된 dropout
        self.dropout2 = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        batch_size, seq_len, total_features = x.shape
        
        weather_data = x[:, :, :4]
        pattern_data = x[:, :, 4].long()
        
        pattern_data = torch.clamp(pattern_data, 0, self.n_patterns - 1)
        
        # Pattern embedding with dropout
        pattern_emb = self.pattern_embed(pattern_data)
        pattern_emb = self.embedding_dropout(pattern_emb)
        
        combined_features = torch.cat([weather_data, pattern_emb], dim=2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # Layer Normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        out = attn_out[:, -1, :]
        
        # FC layers with strong regularization
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out.squeeze()
    
    def train_model(self, train_loader, val_loader=None, epochs=100, lr=0.0005):
        """모델 학습 함수 (개선된 조기 종료)"""
        criterion = nn.MSELoss()
        # Weight decay 추가
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
        
        # ReduceLROnPlateau 스케줄러 사용
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        train_losses = []
        val_losses = []
        
        print(f"총 {epochs} 에포크 학습을 시작합니다...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        patience = 15  # 증가된 patience
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = self(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                
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
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        preds = self(batch_X)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping with best model saving
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Load best model
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                    break
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    epoch_time = time.time() - epoch_start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f} | "
                          f"LR: {current_lr:.6f} | "
                          f"시간: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n학습 완료! 총 소요시간: {total_time/60:.1f}분")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        """모델 예측 함수"""
        self.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                preds = self(batch_X)
                
                preds_np = preds.cpu().numpy()
                actuals_np = batch_y.cpu().numpy()
                
                if preds_np.ndim == 0:
                    predictions.append(preds_np.item())
                    actuals.append(actuals_np.item())
                else:
                    predictions.extend(preds_np)
                    actuals.extend(actuals_np)
        
        return np.array(predictions), np.array(actuals)


class GRU_Pattern(nn.Module):
    """
    과적합 방지가 강화된 GRU 모델
    """
    def __init__(self, weather_feature_dim=4, pattern_embedding_dim=16, hidden_dim=64, num_layers=2, n_patterns=50):
        super(GRU_Pattern, self).__init__()
        
        self.n_patterns = n_patterns
        self.hidden_dim = hidden_dim
        
        self.pattern_embed = nn.Embedding(n_patterns, pattern_embedding_dim)
        self.embedding_dropout = nn.Dropout(0.3)
        
        gru_input_size = weather_feature_dim + pattern_embedding_dim
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=0.4)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, 
                                              batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        batch_size, seq_len, total_features = x.shape
        
        weather_data = x[:, :, :4]
        pattern_data = x[:, :, 4].long()
        
        pattern_data = torch.clamp(pattern_data, 0, self.n_patterns - 1)
        
        pattern_emb = self.pattern_embed(pattern_data)
        pattern_emb = self.embedding_dropout(pattern_emb)
        
        combined_features = torch.cat([weather_data, pattern_emb], dim=2)
        
        gru_out, h_n = self.gru(combined_features)
        
        gru_out = self.layer_norm(gru_out)
        
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        out = attn_out[:, -1, :]
        
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return out.squeeze()
    
    def train_model(self, train_loader, val_loader=None, epochs=100, lr=0.0005):
        """모델 학습 함수"""
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        train_losses = []
        val_losses = []
        
        print(f"총 {epochs} 에포크 학습을 시작합니다...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            self.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                preds = self(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            if val_loader:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        preds = self(batch_X)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                    break
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    epoch_time = time.time() - epoch_start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                          f"Train Loss: {avg_train_loss:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f} | "
                          f"LR: {current_lr:.6f} | "
                          f"시간: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n학습 완료! 총 소요시간: {total_time/60:.1f}분")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        """모델 예측 함수"""
        self.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                preds = self(batch_X)
                
                preds_np = preds.cpu().numpy()
                actuals_np = batch_y.cpu().numpy()
                
                if preds_np.ndim == 0:
                    predictions.append(preds_np.item())
                    actuals.append(actuals_np.item())
                else:
                    predictions.extend(preds_np)
                    actuals.extend(actuals_np)
        
        return np.array(predictions), np.array(actuals)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TimeSeriesDataGenerator:
    """시계열 데이터 생성기"""
    
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.location_encoder = LabelEncoder()
        
    def prepare_sequences(self, df, feature_cols, target_col, location_col):
        """시계열 시퀀스 데이터 생성"""
        print("\n" + "=" * 80)
        print("9. 시계열 시퀀스 데이터 생성")
        print("=" * 80)
        
        df[location_col] = df[location_col].astype(str)
        
        df['교차로ID'] = self.location_encoder.fit_transform(df[location_col])
        
        X_sequences = []
        y_targets = []
        location_ids = []
        
        for location in df['교차로ID'].unique():
            location_data = df[df['교차로ID'] == location].sort_values('일시')
            
            if len(location_data) < self.sequence_length + 1:
                continue
            
            features = location_data[feature_cols].values
            target = location_data[target_col].values
            
            for i in range(len(features) - self.sequence_length):
                X_sequences.append(features[i:i+self.sequence_length])
                y_targets.append(target[i+self.sequence_length])
                location_ids.append(location)
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        locations = np.array(location_ids)
        
        print(f"\n생성된 시퀀스 수: {len(X)}")
        print(f"시퀀스 shape: {X.shape}")
        print(f"타겟 shape: {y.shape}")
        print(f"\n지점별 시퀀스 분포:")
        for loc_id in np.unique(locations):
            loc_name = self.location_encoder.inverse_transform([loc_id])[0]
            count = (locations == loc_id).sum()
            print(f"  {loc_name}: {count} sequences")
        
        return X, y, locations
    
    def normalize_features(self, X_train, X_test):
        """특성 정규화"""
        n_samples, n_timesteps, n_features = X_train.shape
        
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_test_scaled = X_test_scaled.reshape(-1, n_timesteps, n_features)
        
        return X_train_scaled, X_test_scaled


def train_xgboost_stacking(base_predictions_train, y_train, base_predictions_val, y_val, 
                           base_predictions_test, y_test, X_train_features, X_val_features, X_test_features):
    """XGBoost 스태킹 모델 학습 및 평가 (과적합 방지 강화)"""
    print("\n" + "=" * 80)
    print("12. XGBoost 스태킹 모델 학습 (과적합 방지 강화)")
    print("=" * 80)
    
    print("\n특성 결합 중...")
    train_features = np.column_stack([base_predictions_train, X_train_features])
    val_features = np.column_stack([base_predictions_val, X_val_features])
    test_features = np.column_stack([base_predictions_test, X_test_features])
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # XGBoost 하이퍼파라미터 - 과적합 방지 강화
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,           # 트리 수 증가
        learning_rate=0.03,         # 더 낮은 학습률
        max_depth=3,                # 더 얕은 깊이
        min_child_weight=10,        # 리프 노드 최소 샘플 수 증가
        subsample=0.7,              # 낮은 샘플 서브샘플링
        colsample_bytree=0.7,       # 낮은 특성 서브샘플링
        gamma=0.2,                  # 높은 분할 최소 손실 감소
        reg_alpha=0.1,              # 강한 L1 정규화
        reg_lambda=2.0,             # 강한 L2 정규화
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=30,
        enable_categorical=False
    )
    
    print("\nXGBoost 학습 중...")
    xgb_model.fit(
        train_features, 
        y_train, 
        eval_set=[(val_features, y_val)],
        verbose=False
    )
    
    # 예측
    pred_train = xgb_model.predict(train_features)
    pred_val = xgb_model.predict(val_features)
    pred_test = xgb_model.predict(test_features)
    
    # 평가 지표 계산
    train_nmae = calculate_nmae(y_train, pred_train)
    train_r2 = calculate_r2(y_train, pred_train)
    train_mape = calculate_mape(y_train, pred_train)
    train_acc = calculate_accuracy_within_threshold(y_train, pred_train, threshold_percent=10)
    
    val_nmae = calculate_nmae(y_val, pred_val)
    val_r2 = calculate_r2(y_val, pred_val)
    val_mape = calculate_mape(y_val, pred_val)
    val_acc = calculate_accuracy_within_threshold(y_val, pred_val, threshold_percent=10)
    
    test_nmae = calculate_nmae(y_test, pred_test)
    test_r2 = calculate_r2(y_test, pred_test)
    test_mape = calculate_mape(y_test, pred_test)
    test_acc = calculate_accuracy_within_threshold(y_test, pred_test, threshold_percent=10)
    
    print("\n=== XGBoost 스태킹 모델 성능 ===")
    print(f"\n{'Dataset':<10} {'NMAE':<10} {'R²':<10} {'MAPE(%)':<12} {'Accuracy(%)':>15}")
    print("-" * 65)
    print(f"{'Train':<10} {train_nmae:<10.4f} {train_r2:<10.4f} {train_mape:<12.2f} {train_acc:>15.2f}")
    print(f"{'Val':<10} {val_nmae:<10.4f} {val_r2:<10.4f} {val_mape:<12.2f} {val_acc:>15.2f}")
    print(f"{'Test':<10} {test_nmae:<10.4f} {test_r2:<10.4f} {test_mape:<12.2f} {test_acc:>15.2f}")
    
    # 과적합 체크
    print("\n=== 과적합 체크 ===")
    r2_diff = train_r2 - test_r2
    if r2_diff < 0.1:
        print(f"✓ 과적합 없음: Train R² ({train_r2:.4f}) - Test R² ({test_r2:.4f}) = {r2_diff:.4f}")
    elif r2_diff < 0.2:
        print(f"⚠ 경미한 과적합: Train R² ({train_r2:.4f}) - Test R² ({test_r2:.4f}) = {r2_diff:.4f}")
    else:
        print(f"✗ 과적합 발생: Train R² ({train_r2:.4f}) - Test R² ({test_r2:.4f}) = {r2_diff:.4f}")
    
    # KPI 체크
    print("\n=== KPI 체크 ===")
    kpi_threshold = 90.0
    r2_threshold = 0.8
    
    if test_acc >= kpi_threshold:
        print(f"✓ 정확도 KPI 달성: {test_acc:.2f}% (목표: {kpi_threshold}% 이상)")
    else:
        print(f"✗ 정확도 KPI 미달: {test_acc:.2f}% (목표: {kpi_threshold}% 이상)")
    
    if test_r2 >= r2_threshold:
        print(f"✓ R² KPI 달성: {test_r2:.4f} (목표: {r2_threshold} 이상)")
    else:
        print(f"✗ R² KPI 미달: {test_r2:.4f} (목표: {r2_threshold} 이상)")
    
    # 특성 중요도
    feature_names = ['LSTM_예측값', 'GRU_예측값'] + [f'원본특성_{i+1}' for i in range(train_features.shape[1] - 2)]
    importance_dict = dict(zip(feature_names, xgb_model.feature_importances_))
    
    print("\n특성 중요도 (Top 10):")
    for i, (feature, importance) in enumerate(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return xgb_model, pred_test, {
        'NMAE': test_nmae,
        'R2': test_r2,
        'MAPE': test_mape,
        'Accuracy': test_acc
    }

def evaluate_model(y_true, y_pred, model_name):
    """모델 성능 평가"""
    nmae = calculate_nmae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    accuracy = calculate_accuracy_within_threshold(y_true, y_pred, threshold_percent=10)
    
    print(f"\n{model_name} 성능:")
    print(f"  NMAE:        {nmae:.4f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  MAPE:        {mape:.2f}%")
    print(f"  Accuracy:    {accuracy:.2f}% (±10% 오차 이내)")
    
    return {
        'NMAE': nmae,
        'R2': r2,
        'MAPE': mape,
        'Accuracy': accuracy
    }


def main():
    """메인 실행 함수"""
    
    print("\n")
    print("=" * 80)
    print("계절별 기상 패턴 클러스터링 기반 도로 열위험도 예측 시스템 (과적합 방지)")
    print("=" * 80)
    
    # 1. 태양광 발전 데이터 로드 및 전처리
    solar_preprocessor = SolarDataPreprocessor()
    
    solar_csv = './dataset/output_by_region/광주.csv'
    solar_df = solar_preprocessor.load_solar_data(solar_csv)
    
    solar_df = solar_preprocessor.handle_missing_outliers(solar_df)
    solar_df = solar_preprocessor.add_temporal_features(solar_df)
    
    # 2. 계절별 K-means 클러스터링
    seasonal_clustering = SeasonalWeatherPatternClustering(n_clusters=5)
    solar_df = seasonal_clustering.fit_predict(solar_df)
    
    # 3. 도로 데이터 로드 및 전처리
    road_preprocessor = RoadDataPreprocessor()
    
    road_csv = './dataset/양산교차로.csv'
    road_df = road_preprocessor.load_and_parse_csv(road_csv)
    
    road_df = road_preprocessor.handle_missing_outliers(road_df)
    road_df = road_preprocessor.add_temporal_features(road_df)
    
    # 4. 도로 데이터에 기상 패턴 매핑
    road_df = road_preprocessor.merge_with_weather_patterns(road_df, solar_df, seasonal_clustering)
    
    # 5. 시계열 데이터 준비
    feature_cols = [
        '기온', '습도_filled', '시간_sin', '시간_cos', '기상패턴'
    ]
    
    target_col = '기온'
    location_col = '지점명'
    
    data_gen = TimeSeriesDataGenerator(sequence_length=24)
    X, y, locations = data_gen.prepare_sequences(road_df, feature_cols, target_col, location_col)
    
    # 6. 학습/검증/테스트 데이터 분할
    print("\n" + "=" * 80)
    print("10. 학습/검증/테스트 데이터 분할")
    print("=" * 80)
    
    train_ratio = 0.7
    val_ratio = 0.15
    
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    print(f"\n학습 데이터: {len(X_train)} sequences")
    print(f"검증 데이터: {len(X_val)} sequences")
    print(f"테스트 데이터: {len(X_test)} sequences")
    
    # 특성 정규화
    X_train_scaled, X_test_scaled = data_gen.normalize_features(X_train, X_test)
    X_val_scaled, _ = data_gen.normalize_features(X_val, X_val)
    
    # DataLoader 생성
    batch_size = 64  # 더 큰 배치 사이즈로 일반화 향상
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 7. LSTM_Pattern 모델 학습
    print("\n" + "=" * 80)
    print("11-1. LSTM_Pattern 모델 학습 ")
    print("=" * 80)
    
    lstm_model = LSTM_Pattern(
        weather_feature_dim=4,
        pattern_embedding_dim=16,
        hidden_dim=64,
        num_layers=2,
        n_patterns=50
    ).to(device)
    
    lstm_train_losses, lstm_val_losses = lstm_model.train_model(
        train_loader, val_loader, epochs=100, lr=0.0005
    )
    
    lstm_pred_train, _ = lstm_model.predict(train_loader)
    lstm_pred_val, _ = lstm_model.predict(val_loader)
    lstm_pred_test, _ = lstm_model.predict(test_loader)
    
    lstm_results = evaluate_model(y_test, lstm_pred_test, "LSTM_Pattern")
    
    # 8. GRU_Pattern 모델 학습
    print("\n" + "=" * 80)
    print("11-2. GRU_Pattern 모델 학습")
    print("=" * 80)
    
    gru_model = GRU_Pattern(
        weather_feature_dim=4,
        pattern_embedding_dim=16,
        hidden_dim=64,
        num_layers=2,
        n_patterns=50
    ).to(device)
    
    gru_train_losses, gru_val_losses = gru_model.train_model(
        train_loader, val_loader, epochs=100, lr=0.0005
    )
    
    gru_pred_train, _ = gru_model.predict(train_loader)
    gru_pred_val, _ = gru_model.predict(val_loader)
    gru_pred_test, _ = gru_model.predict(test_loader)
    
    gru_results = evaluate_model(y_test, gru_pred_test, "GRU_Pattern")
    
    # 9. XGBoost 스태킹용 추가 특성 준비
    X_train_last = X_train_scaled[:, -1, :4]
    X_val_last = X_val_scaled[:, -1, :4]
    X_test_last = X_test_scaled[:, -1, :4]
    
    base_predictions_train = np.column_stack([lstm_pred_train, gru_pred_train])
    base_predictions_val = np.column_stack([lstm_pred_val, gru_pred_val])
    base_predictions_test = np.column_stack([lstm_pred_test, gru_pred_test])
    
    # 10. XGBoost 스태킹
    xgb_model, xgb_pred_test, xgb_results = train_xgboost_stacking(
        base_predictions_train, y_train,
        base_predictions_val, y_val,
        base_predictions_test, y_test,
        X_train_last, X_val_last, X_test_last
    )
    
    # 11. 최종 결과 요약
    print("\n" + "=" * 80)
    print("13. 최종 성능 요약")
    print("=" * 80)
    print(f"\n{'모델':<30} {'NMAE':<12} {'R²':<12} {'MAPE(%)':<12} {'Accuracy(%)':>15}")
    print("-" * 85)
    print(f"{'LSTM_Pattern':<30} {lstm_results['NMAE']:<12.4f} {lstm_results['R2']:<12.4f} "
          f"{lstm_results['MAPE']:<12.2f} {lstm_results['Accuracy']:>15.2f}")
    print(f"{'GRU_Pattern':<30} {gru_results['NMAE']:<12.4f} {gru_results['R2']:<12.4f} "
          f"{gru_results['MAPE']:<12.2f} {gru_results['Accuracy']:>15.2f}")
    print(f"{'XGBoost Stacking':<30} {xgb_results['NMAE']:<12.4f} {xgb_results['R2']:<12.4f} "
          f"{xgb_results['MAPE']:<12.2f} {xgb_results['Accuracy']:>15.2f}")
    
    # KPI 최종 확인
    print("\n" + "=" * 80)
    print("14. KPI 최종 확인")
    print("=" * 80)
    kpi_target = 90.0
    r2_target = 0.8
    
    best_model = max(
        [('LSTM_Pattern', lstm_results['Accuracy'], lstm_results['R2']),
         ('GRU_Pattern', gru_results['Accuracy'], gru_results['R2']),
         ('XGBoost Stacking', xgb_results['Accuracy'], xgb_results['R2'])],
        key=lambda x: (x[1], x[2])
    )
    
    print(f"\n최고 성능 모델: {best_model[0]}")
    print(f"  Accuracy: {best_model[1]:.2f}%")
    print(f"  R²: {best_model[2]:.4f}")
    
    if best_model[1] >= kpi_target and best_model[2] >= r2_target:
        print(f"✓ 전체 KPI 달성!")
    else:
        if best_model[1] < kpi_target:
            print(f"✗ 정확도 미달: {kpi_target - best_model[1]:.2f}%p 부족")
        if best_model[2] < r2_target:
            print(f"✗ R² 미달: {r2_target - best_model[2]:.4f} 부족")
    
    # 12. 모델 저장
    print("\n" + "=" * 80)
    print("15. 모델 저장")
    print("=" * 80)
    
    torch.save(lstm_model.state_dict(), './heatmap_prediction/lstm_pattern_improved.pth')
    torch.save(gru_model.state_dict(), './heatmap_prediction/gru_pattern_improved.pth')
    print("\n모델 저장 완료:")
    print("- ./heatmap_prediction/lstm_pattern_improved.pth")
    print("- ./heatmap_prediction/gru_pattern_improved.pth")
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()