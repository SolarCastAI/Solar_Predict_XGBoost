"""
저장된 모델을 로드하여 새로운 데이터에 대해 예측을 수행하는 예제
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import os
import xgboost as xgb

# GPU/CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 사용 중인 디바이스: {device}")


# === 모델 클래스 정의 (학습 코드와 동일) ===
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


# === 모델 로드 함수 ===
def load_trained_models(model_dir='./saved_models', timestamp=None):
    """
    저장된 모델과 스케일러 로드
    
    Args:
        model_dir: 모델이 저장된 디렉토리
        timestamp: 특정 시점의 모델을 로드하려면 타임스탬프 지정
                  None이면 최신 모델 로드
    
    Returns:
        dict: 로드된 모델, 스케일러, 메타데이터를 포함하는 딕셔너리
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
        print(f"✅ 최신 모델 타임스탬프: {timestamp}")
    
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
    print(f"   - Input size: {lstm_config['input_size']}")
    print(f"   - Hidden size: {lstm_config['hidden_size']}")
    print(f"   - Num layers: {lstm_config['num_layers']}")
    
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
    print(f"   - Input size: {gru_config['input_size']}")
    print(f"   - Hidden size: {gru_config['hidden_size']}")
    print(f"   - Num layers: {gru_config['num_layers']}")
    
    # XGBoost 모델 로드
    xgb_path = os.path.join(model_dir, f'xgboost_stacking_{timestamp}.json')
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    print(f"✅ XGBoost 스태킹 모델 로드: {xgb_path}")
    
    # 모델 성능 정보 출력
    print(f"\n{'='*80}")
    print("모델 성능 정보")
    print(f"{'='*80}")
    print("\nLSTM 모델:")
    for metric, value in lstm_checkpoint.get('metrics', {}).items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
    
    print("\nGRU 모델:")
    for metric, value in gru_checkpoint.get('metrics', {}).items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
    
    print("\nXGBoost 스태킹 모델:")
    for metric, value in metadata.get('stacked_metrics', {}).items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
    
    print(f"\n{'='*80}")
    print("✨ 모든 모델이 성공적으로 로드되었습니다!")
    print(f"{'='*80}")
    
    return {
        'lstm_model': lstm_model,
        'gru_model': gru_model,
        'xgb_model': xgb_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metadata': metadata
    }


# === 예측 함수 ===
def predict_solar_generation(new_data, models_dict, sequence_length=24):
    """
    새로운 데이터에 대해 태양광 발전량 예측
    
    Args:
        new_data: pandas DataFrame 또는 numpy array (특성 데이터)
                  shape: (n_samples, n_features)
        models_dict: load_trained_models()에서 반환된 딕셔너리
        sequence_length: 시퀀스 길이 (기본값: 24)
    
    Returns:
        dict: LSTM, GRU, 스태킹 모델의 예측 결과
    """
    # 모델과 스케일러 추출
    lstm_model = models_dict['lstm_model']
    gru_model = models_dict['gru_model']
    xgb_model = models_dict['xgb_model']
    scaler_X = models_dict['scaler_X']
    scaler_y = models_dict['scaler_y']
    metadata = models_dict['metadata']
    
    # 특성 컬럼 정보
    feature_cols = metadata['feature_columns']
    expected_features = len(feature_cols)
    
    print(f"\n예측 데이터 정보:")
    print(f"  예상 특성 개수: {expected_features}")
    print(f"  예상 특성 목록: {feature_cols}")
    
    # 데이터 형식 변환 및 특성 순서 보장
    if isinstance(new_data, pd.DataFrame):
        print(f"  입력 데이터 shape: {new_data.shape}")
        print(f"  입력 데이터 컬럼: {list(new_data.columns)}")
        
        # 메타데이터의 feature 순서대로 정렬
        try:
            new_data = new_data[feature_cols].values
            print(f"  ✅ 특성 컬럼 순서 정렬 완료")
        except KeyError as e:
            missing_cols = set(feature_cols) - set(new_data.columns)
            print(f"  ❌ 오류: 다음 컬럼이 데이터에 없습니다: {missing_cols}")
            raise ValueError(f"필수 특성 컬럼이 없습니다: {missing_cols}")
    else:
        print(f"  입력 데이터 shape: {new_data.shape}")
        if new_data.shape[1] != expected_features:
            raise ValueError(
                f"입력 데이터의 특성 개수({new_data.shape[1]})가 "
                f"예상과 다릅니다(예상: {expected_features})"
            )
    
    # 데이터가 시퀀스 길이보다 작으면 에러
    if len(new_data) < sequence_length:
        raise ValueError(f"데이터 길이({len(new_data)})가 시퀀스 길이({sequence_length})보다 작습니다.")
    
    # 결측값 처리 - numpy array로 변환 후 처리
    new_data_imputed = new_data.copy()
    
    print(f"\n결측값 처리:")
    # 각 컬럼의 결측값 확인 및 처리
    for col_idx in range(new_data_imputed.shape[1]):
        col_data = new_data_imputed[:, col_idx]
        nan_count = np.sum(np.isnan(col_data))
        
        if nan_count > 0:
            col_name = feature_cols[col_idx] if col_idx < len(feature_cols) else f"컬럼 {col_idx}"
            
            if np.all(np.isnan(col_data)):
                # 완전히 결측인 컬럼은 0으로 채움
                new_data_imputed[:, col_idx] = 0
                print(f"  ⚠️  {col_name}: 모든 값이 결측 → 0으로 채움")
            else:
                # 부분적으로 결측인 경우 평균값으로 채움
                col_mean = np.nanmean(col_data)
                new_data_imputed[:, col_idx] = np.where(
                    np.isnan(col_data), 
                    col_mean, 
                    col_data
                )
                print(f"  ℹ️  {col_name}: {nan_count}개 결측값 → 평균({col_mean:.2f})으로 채움")
    
    # 최종 shape 확인
    print(f"\n전처리 완료:")
    print(f"  최종 데이터 shape: {new_data_imputed.shape}")
    print(f"  스케일러 예상 특성: {scaler_X.n_features_in_}")
    
    # 데이터 스케일링
    try:
        new_data_scaled = scaler_X.transform(new_data_imputed)
        print(f"  ✅ 스케일링 완료")
    except ValueError as e:
        print(f"  ❌ 스케일링 오류: {e}")
        print(f"     입력 데이터 shape: {new_data_imputed.shape}")
        print(f"     스케일러 예상 특성: {scaler_X.n_features_in_}")
        raise
    
    # 시퀀스 데이터 생성
    X_sequences = []
    for i in range(len(new_data_scaled) - sequence_length + 1):
        X_sequences.append(new_data_scaled[i:i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    
    # PyTorch 텐서로 변환
    X_tensor = torch.FloatTensor(X_sequences).to(device)
    
    # LSTM 예측
    lstm_model.eval()
    with torch.no_grad():
        lstm_predictions_scaled = lstm_model(X_tensor).cpu().numpy()
    
    lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled)
    
    # GRU 예측
    gru_model.eval()
    with torch.no_grad():
        gru_predictions_scaled = gru_model(X_tensor).cpu().numpy()
    
    gru_predictions = scaler_y.inverse_transform(gru_predictions_scaled)
    
    # 스태킹 모델용 특성 생성
    X_stacked = np.hstack([
        lstm_predictions.reshape(-1, 1),
        gru_predictions.reshape(-1, 1)
    ])
    
    # XGBoost 스태킹 예측
    stacked_predictions = xgb_model.predict(X_stacked).reshape(-1, 1)
    
    return {
        'lstm_predictions': lstm_predictions.flatten(),
        'gru_predictions': gru_predictions.flatten(),
        'stacked_predictions': stacked_predictions.flatten(),
        'n_predictions': len(stacked_predictions)
    }


# === 사용 예제 ===
if __name__ == "__main__":
    try:
        # 1. 저장된 모델 로드
        print("\n" + "="*80)
        print("저장된 모델 로드 예제")
        print("="*80)
        
        models = load_trained_models(model_dir='./saved_models')
        
        # 2. 새로운 데이터 로드 (예제)
        print("\n새로운 데이터로 예측 수행...")
        
        # 실제 사용 시에는 새로운 CSV 파일이나 데이터를 로드
        data_path = "./dataset/jeju_solar_utf8.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
            print("테스트용 더미 데이터를 생성합니다...")
            
            # 더미 데이터 생성
            feature_cols = models['metadata']['feature_columns']
            n_samples = 50
            
            test_data = pd.DataFrame(
                np.random.randn(n_samples, len(feature_cols)),
                columns=feature_cols
            )
        else:
            df = pd.read_csv(data_path)
            
            # 타겟 컬럼 제외
            target_col = '태양광 발전량(MWh)'
            feature_cols = models['metadata']['feature_columns']
            
            # 필수 컬럼 확인
            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols:
                print(f"❌ 오류: 다음 컬럼이 데이터에 없습니다: {missing_cols}")
                print(f"데이터 컬럼: {list(df.columns)}")
                raise ValueError(f"필수 특성 컬럼이 없습니다: {missing_cols}")
            
            # 테스트용으로 마지막 50개 데이터만 사용
            test_data = df[feature_cols].tail(50).copy()
        
        print(f"테스트 데이터 shape: {test_data.shape}")
        print(f"테스트 데이터 컬럼: {list(test_data.columns)}")
        
        # 3. 예측 수행
        predictions = predict_solar_generation(
            new_data=test_data,
            models_dict=models,
            sequence_length=24
        )
        
        # 4. 결과 출력
        print(f"\n{'='*80}")
        print("예측 결과")
        print(f"{'='*80}")
        print(f"총 예측 개수: {predictions['n_predictions']}")
        print(f"\nLSTM 예측 (처음 5개):")
        print(predictions['lstm_predictions'][:5])
        print(f"\nGRU 예측 (처음 5개):")
        print(predictions['gru_predictions'][:5])
        print(f"\n스태킹 예측 (처음 5개):")
        print(predictions['stacked_predictions'][:5])
        
        # 5. 예측 결과를 DataFrame으로 저장
        results_df = pd.DataFrame({
            'LSTM_Prediction': predictions['lstm_predictions'],
            'GRU_Prediction': predictions['gru_predictions'],
            'Stacked_Prediction': predictions['stacked_predictions']
        })
        
        output_path = './predictions_output.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 예측 결과가 저장되었습니다: {output_path}")
        
        # 6. 통계 정보
        print(f"\n{'='*80}")
        print("예측 통계")
        print(f"{'='*80}")
        print(f"\nLSTM 예측:")
        print(f"  평균: {predictions['lstm_predictions'].mean():.4f} MWh")
        print(f"  최소: {predictions['lstm_predictions'].min():.4f} MWh")
        print(f"  최대: {predictions['lstm_predictions'].max():.4f} MWh")
        print(f"  표준편차: {predictions['lstm_predictions'].std():.4f} MWh")
        
        print(f"\nGRU 예측:")
        print(f"  평균: {predictions['gru_predictions'].mean():.4f} MWh")
        print(f"  최소: {predictions['gru_predictions'].min():.4f} MWh")
        print(f"  최대: {predictions['gru_predictions'].max():.4f} MWh")
        print(f"  표준편차: {predictions['gru_predictions'].std():.4f} MWh")
        
        print(f"\n스태킹 예측:")
        print(f"  평균: {predictions['stacked_predictions'].mean():.4f} MWh")
        print(f"  최소: {predictions['stacked_predictions'].min():.4f} MWh")
        print(f"  최대: {predictions['stacked_predictions'].max():.4f} MWh")
        print(f"  표준편차: {predictions['stacked_predictions'].std():.4f} MWh")
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("먼저 모델을 학습하고 저장해야 합니다.")
        print("solar_prediction_with_save.py를 실행하여 모델을 학습하세요.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()