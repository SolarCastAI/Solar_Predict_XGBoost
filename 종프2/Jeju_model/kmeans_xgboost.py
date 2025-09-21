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
data_path = "./dataset/jeju_solar_utf8.csv"
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


def xgboost_pattern_model(X_train, y_train, X_val, y_val, X_test, y_test, plotting=False):
    """
    패턴 정보를 활용한 XGBoost 모델
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
    
    print("XGBoost 패턴 모델 학습 중...")
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
                       '기온×일사량', '일조×일사량', '무강수여부']
    
    importance_dict = dict(zip(feature_names, xgb_regressor.feature_importances_))
    print("\n=== XGBoost 패턴 모델 특성 중요도 ===")
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
        plt.title(f"XGBoost 패턴 모델 - MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.3f}%")
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
        
        # 3. 최종 특성 벡터 생성 (기상 데이터 + 패턴 특성)
        X_combined = np.hstack([features, pattern_features])
        y = targets
        
        print(f"\n최종 특성 벡터 크기: {X_combined.shape}")
        print(f"타겟 벡터 크기: {y.shape}")
        
        # 4. 데이터셋 시간순 분할 (70% train, 15% validation, 15% test)
        print("\n시간순으로 데이터 분할...")
        total_size = len(X_combined)
        train_end = int(total_size * 0.7)
        val_end = int(total_size * 0.85)

        X_train, y_train = X_combined[:train_end], y[:train_end]
        X_val, y_val = X_combined[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X_combined[val_end:], y[val_end:]
        
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # 5. XGBoost 모델 학습 및 평가
        print("\n" + "="*60)
        print("XGBoost 패턴 모델 학습 시작...")
        print("="*60)
        mae_xgb, rmse_xgb, mape_xgb, xgb_model = xgboost_pattern_model(
            X_train, y_train, X_val, y_val, X_test, y_test, plotting=True
        )
        
        print(f"\n=== 최종 XGBoost 패턴 모델 성능 ===")
        print(f"MAE: {mae_xgb:.4f}")
        print(f"RMSE: {rmse_xgb:.4f}")
        print(f"MAPE: {mape_xgb:.4f}%")
        
        # 6. 패턴별 예측 성능 분석
        print("\n=== 패턴별 예측 성능 분석 ===")
        test_patterns = pattern_features[val_end:, 0].astype(int)  # 테스트셋의 패턴 라벨
        pred_test = xgb_model.predict(X_test)
        
        for pattern_id in range(pattern_extractor.n_patterns):
            mask = test_patterns == pattern_id
            if mask.sum() > 0:
                pattern_mae = mean_absolute_error(y_test[mask], pred_test[mask])
                pattern_rmse = calculate_rmse(y_test[mask], pred_test[mask])
                pattern_mape = calculate_mape(y_test[mask], pred_test[mask])
                
                print(f"패턴 {pattern_id+1}: 데이터수={mask.sum():4d}, "
                      f"MAE={pattern_mae:.4f}, RMSE={pattern_rmse:.4f}, MAPE={pattern_mape:.2f}%")
        
        print(f"\n{'='*60}")
        print("K-means 패턴 추출 + XGBoost 모델 학습 완료")
        print(f"{'='*60}")
        
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()