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

# 사용 예시를 위한 device 설정
data_path = "./dataset/jeju_solar_utf8.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")

print(f"device : {device}")

plt.style.use('seaborn-v0_8-whitegrid')

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


class ResidualBlock(nn.Module):
    """
    잔차 연결을 포함한 CNN 블록
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - 차원이 다를 경우 1x1 conv로 맞춤
        self.skip_connection = None
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection 적용
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        out += identity  # 잔차 연결
        out = self.relu(out)
        
        return out


class BidirectionalAttention(nn.Module):
    """
    양방향 self-attention 메커니즘
    """
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(BidirectionalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value 변환 레이어
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # 출력 변환 레이어
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Position-wise feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Scaled Dot-Product Attention 계산
        """
        batch_size, seq_len, _ = query.size()
        
        # Multi-head로 reshape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores 계산
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Mask 적용 (필요한 경우)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 적용
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted values 계산
        context = torch.matmul(attention_weights, value)
        
        # Multi-head 결과 concat
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        return context, attention_weights
    
    def forward(self, x, mask=None):
        """
        양방향 attention forward pass
        """
        batch_size, seq_len, _ = x.size()
        
        # Skip connection을 위한 residual
        residual = x
        
        # Query, Key, Value 계산
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        
        # Self-attention 적용
        context, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask
        )
        
        # 출력 변환 및 잔차 연결
        output = self.output_linear(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        # Position-wise feedforward network with residual connection
        ffn_output = self.ffn(output)
        output = self.layer_norm2(output + ffn_output)
        
        return output, attention_weights


class Enhanced_CNN_LSTM_Pattern(nn.Module):
    """
    양방향 attention과 잔차 연결을 적용한 향상된 CNN + LSTM 모델
    """
    def __init__(self, input_dim=5, seq_len=24, hidden_dim=128, num_layers=2, 
                 n_patterns=5, num_attention_heads=8, dropout=0.2):
        super(Enhanced_CNN_LSTM_Pattern, self).__init__()
        
        self.n_patterns = n_patterns
        self.hidden_dim = hidden_dim
        
        # Pattern Embedding Layer (더 큰 차원으로 확장)
        self.pattern_embed = nn.Embedding(n_patterns, 16)
        self.pattern_dropout = nn.Dropout(0.1)
        
        # Enhanced CNN layers with Residual Blocks
        self.conv_input = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=1)
        
        # 여러 개의 잔차 블록
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(32, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128, kernel_size=3, padding=1),
        ])
        
        # Adaptive pooling for flexible sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(seq_len//2)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128+16,  # CNN output + pattern embedding
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Bidirectional Attention
        self.bidirectional_attention = BidirectionalAttention(
            hidden_dim * 2,  # Bidirectional LSTM outputs 2*hidden_dim
            num_heads=num_attention_heads, 
            dropout=dropout
        )
        
        # Additional Attention layers
        self.attention_layers = nn.ModuleList([
            BidirectionalAttention(hidden_dim * 2, num_attention_heads, dropout)
            for _ in range(2)  # Multiple attention layers
        ])
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced output layers with residual connections
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Linear(hidden_dim // 4, 1)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 4)
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        가중치 초기화
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
    
    def forward(self, x):
        batch_size, seq_len, total_features = x.shape
        
        # 기상 데이터와 패턴 데이터 분리
        weather_data = x[:, :, :4]  # 기온, 강수량, 일조, 일사량
        pattern_data = x[:, :, 4].long()  # 패턴 레이블
        
        # CNN for weather data with residual connections
        weather_data = weather_data.permute(0, 2, 1)  # (batch, features, seq_len)
        conv_out = self.conv_input(weather_data)
        
        # Apply residual blocks
        for residual_block in self.residual_blocks:
            conv_out = residual_block(conv_out)
        
        # Adaptive pooling
        conv_out = self.adaptive_pool(conv_out)  # (batch, 128, seq_len//2)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, seq_len//2, 128)
        
        # Pattern embedding
        pattern_emb = self.pattern_embed(pattern_data)  # (batch, seq_len, 16)
        pattern_emb = self.pattern_dropout(pattern_emb)
        
        # Adjust sequence length for pattern embedding to match CNN output
        target_seq_len = conv_out.size(1)
        if pattern_emb.size(1) != target_seq_len:
            # Adaptive pooling for pattern embeddings
            pattern_emb = pattern_emb.permute(0, 2, 1)  # (batch, 16, seq_len)
            pattern_emb = nn.functional.adaptive_avg_pool1d(pattern_emb, target_seq_len)
            pattern_emb = pattern_emb.permute(0, 2, 1)  # (batch, target_seq_len, 16)
        
        # Combine CNN output with pattern embedding
        combined_features = torch.cat([conv_out, pattern_emb], dim=2)
        
        # Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # Apply multiple bidirectional attention layers
        attn_out = lstm_out
        attention_weights_list = []
        
        for attention_layer in self.attention_layers:
            attn_out, attn_weights = attention_layer(attn_out)
            attention_weights_list.append(attn_weights)
        
        # Feature fusion
        # Global average pooling + max pooling combination
        avg_pooled = torch.mean(attn_out, dim=1)  # (batch, hidden_dim*2)
        max_pooled, _ = torch.max(attn_out, dim=1)  # (batch, hidden_dim*2)
        
        # Combine different pooling strategies
        combined_features = avg_pooled + max_pooled
        fused_features = self.feature_fusion(combined_features)
        
        # Enhanced output layers with residual connections
        out = fused_features
        for i, (linear_layer, batch_norm) in enumerate(zip(self.output_layers[:-1], self.batch_norms)):
            residual = out if out.size(-1) == linear_layer.out_features else None
            
            out = linear_layer(out)
            out = batch_norm(out)
            out = self.relu(out)
            out = self.dropout(out)
            
            # Residual connection if dimensions match
            if residual is not None and residual.size(-1) == out.size(-1):
                out = out + residual
        
        # Final output layer
        out = self.output_layers[-1](out)
        
        return out.squeeze(), attention_weights_list
    
    def train_model(self, train_loader, val_loader=None, epochs=100, lr=0.001, 
                   weight_decay=1e-4, patience=15):
        """
        향상된 모델 학습 함수
        """
        # Loss function with label smoothing
        criterion = nn.SmoothL1Loss()  # Huber Loss - 이상값에 더 강건
        
        # Optimizer with different learning rates for different parts
        optimizer = optim.AdamW([
            {'params': self.pattern_embed.parameters(), 'lr': lr * 0.1},
            {'params': self.residual_blocks.parameters(), 'lr': lr * 0.5},
            {'params': self.lstm.parameters(), 'lr': lr},
            {'params': self.attention_layers.parameters(), 'lr': lr * 1.5},
            {'params': self.output_layers.parameters(), 'lr': lr * 0.8}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=lr*0.01
        )
        
        train_losses = []
        val_losses = []
        
        print(f"Enhanced CNN+LSTM 모델 학습 시작 - 총 {epochs} 에포크")
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        
        start_time = time.time()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.train()
            train_loss = 0
            train_batches = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                preds, attention_weights = self(batch_X)
                loss = criterion(preds, batch_y)
                
                # Gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        preds, _ = self(batch_X)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 베스트 모델 저장 (선택사항)
                    # torch.save(self.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Learning rate scheduling
                scheduler.step()
                
                # 진행상황 출력 (매 5 에포크마다)
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    epoch_time = time.time() - epoch_start_time
                    elapsed_time = time.time() - start_time
                    remaining_epochs = epochs - epoch - 1
                    estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs
                    
                    current_lr = scheduler.get_last_lr()[0]
                    
                    print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                          f"Train Loss: {avg_train_loss:.6f} | "
                          f"Val Loss: {avg_val_loss:.6f} | "
                          f"LR: {current_lr:.8f} | "
                          f"시간: {epoch_time:.1f}s | "
                          f"남은시간: {estimated_remaining/60:.1f}분")
        
        total_time = time.time() - start_time
        print(f"\n학습 완료! 총 소요시간: {total_time/60:.1f}분")
        print(f"최고 검증 손실: {best_val_loss:.6f}")
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        """
        향상된 예측 함수 - attention weights도 함께 반환
        """
        self.eval()
        predictions = []
        actuals = []
        attention_weights_batch = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                preds, attn_weights = self(batch_X)
                
                # Convert tensors to numpy arrays
                preds_np = preds.cpu().numpy()
                actuals_np = batch_y.cpu().numpy()
                
                # Handle single-item batches
                if preds_np.ndim == 0:
                    predictions.append(preds_np.item())
                    actuals.append(actuals_np.item())
                else:
                    predictions.extend(preds_np)
                    actuals.extend(actuals_np)
                
                # Store attention weights for analysis
                attention_weights_batch.append([aw.cpu().numpy() for aw in attn_weights])
        
        return np.array(predictions), np.array(actuals), attention_weights_batch


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
    CNN+LSTM 예측을 포함한 스태킹 XGBoost 모델
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
                      ['CNN_LSTM_예측값'] # 스태킹 특성 이름 추가
    
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
        plt.ylabel("DC Power")
        plt.title(f"XGBoost Stacked Model - MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}%")
        plt.legend()
        
        # 산점도
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
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
        residuals = y_test - pred_test
        plt.scatter(pred_test, residuals, alpha=0.5)
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
        
        # 기존 모델 대신 향상된 모델 사용
        model = Enhanced_CNN_LSTM_Pattern(
            input_dim=5, 
            seq_len=seq_len, 
            hidden_dim=128, 
            num_layers=2, 
            n_patterns=5,
            num_attention_heads=8,
            dropout=0.2
        )
        model.to(device)
        print("\n" + "="*60)
        print("CNN+LSTM 모델 학습 시작...")
        print("="*60)
        train_losses, val_losses = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            lr=0.001
        )

        # 6. CNN+LSTM 모델 성능 평가 - 오류 수정: 3개의 반환값을 처리
        print("\nCNN+LSTM 테스트 예측 및 평가 중...")
        predictions_scaled, actuals_scaled, attention_weights_batch = model.predict(test_loader)
        predictions_original = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        actuals_original = target_scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).flatten()
        
        mae_cnn_lstm = mean_absolute_error(actuals_original, predictions_original)
        rmse_cnn_lstm = calculate_rmse(actuals_original, predictions_original)
        mape_cnn_lstm = calculate_mape(actuals_original, predictions_original)
        
        print(f"\n=== CNN+LSTM 모델 성능 평가 ===")
        print(f"MAE: {mae_cnn_lstm:.4f}")
        print(f"RMSE: {rmse_cnn_lstm:.4f}")
        print(f"MAPE: {mape_cnn_lstm:.4f}%")
        
        # 7. 스태킹을 위한 예측값 생성 (XGBoost의 새로운 특성)
        print("\n" + "="*60)
        print("스태킹을 위한 CNN+LSTM 예측값 생성 중...")
        print("="*60)
        # 순서 유지를 위해 shuffle=False 로 DataLoader 재생성
        train_loader_stack = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader_stack = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        cnn_preds_train_scaled, _, _ = model.predict(train_loader_stack)
        cnn_preds_val_scaled, _, _ = model.predict(val_loader_stack)
        # 테스트 예측값은 이미 계산됨 (predictions_scaled)
        
        cnn_preds_train = target_scaler.inverse_transform(cnn_preds_train_scaled.reshape(-1, 1))
        cnn_preds_val = target_scaler.inverse_transform(cnn_preds_val_scaled.reshape(-1, 1))
        cnn_preds_test = predictions_original.reshape(-1, 1) # 이미 원본 스케일

        # 8. XGBoost 학습을 위한 데이터 준비
        # CNN+LSTM 예측값과 원래 특성들을 결합
        # 시퀀스 생성으로 인해 데이터 길이가 줄어든 것을 반영 (앞부분 seq_len 만큼 제거)
        X_train_xgb = np.hstack([features_train[seq_len:], pattern_train[seq_len:], cnn_preds_train])
        y_train_xgb = targets_train[seq_len:]
        
        X_val_xgb = np.hstack([features_val[seq_len:], pattern_val[seq_len:], cnn_preds_val])
        y_val_xgb = targets_val[seq_len:]
        
        X_test_xgb = np.hstack([features_test[seq_len:], pattern_test[seq_len:], cnn_preds_test])
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
        print(f"CNN+LSTM         : MAE={mae_cnn_lstm:.4f}, RMSE={rmse_cnn_lstm:.4f}, MAPE={mape_cnn_lstm:.2f}%")
        print(f"XGBoost (Stacked): MAE={mae_xgb:.4f}, RMSE={rmse_xgb:.4f}, MAPE={mape_xgb:.2f}%")
        
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