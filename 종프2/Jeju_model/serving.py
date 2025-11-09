"""
실시간 태양광 발전량 예측 시스템
- 현재 시점(2025-11-10) 기준 예측
- 24시간, 48시간, 72시간 이후 태양광 발전량 예측
- 일자별 태양광 발전량 MWh 예측
- 누적 발전량 표시
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
import os
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
    """저장된 모델과 스케일러 로드"""
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
    
    print(f"\n✨ 모든 모델이 성공적으로 로드되었습니다!")
    
    return {
        'lstm_model': lstm_model,
        'gru_model': gru_model,
        'xgb_model': xgb_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metadata': metadata
    }


# === 현재 시점 데이터 준비 함수 ===
def prepare_current_data(data_path, current_datetime, models_dict, hours_needed=96):
    """
    현재 시점까지의 데이터를 준비
    
    Args:
        data_path: 학습 데이터 경로
        current_datetime: 현재 시점 (datetime 객체)
        models_dict: 로드된 모델 딕셔너리
        hours_needed: 필요한 데이터 시간 수 (72시간 예측 + 시퀀스 길이)
    
    Returns:
        DataFrame: 현재 시점까지의 데이터
    """
    print(f"\n{'='*80}")
    print(f"📊 현재 시점 데이터 준비 중... (기준: {current_datetime.strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*80}")
    
    feature_cols = models_dict['metadata']['feature_columns']
    
    if os.path.exists(data_path):
        print(f"✅ 데이터 파일 발견: {data_path}")
        df = pd.read_csv(data_path)
        
        # datetime 컬럼 확인 및 변환
        datetime_col = None
        for col in ['datetime', 'Datetime', 'date', 'Date', '시간', '일시']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            df['datetime'] = pd.to_datetime(df[datetime_col])
        else:
            # datetime 컬럼이 없으면 첫 번째 열을 시간으로 추정
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
        
        # 현재 시점 이전 데이터만 필터링
        df_filtered = df[df['datetime'] <= current_datetime].copy()
        
        if len(df_filtered) == 0:
            print(f"⚠️ 현재 시점({current_datetime}) 이전 데이터가 없습니다.")
            print(f"데이터 범위: {df['datetime'].min()} ~ {df['datetime'].max()}")
            print("가장 최근 데이터를 사용합니다.")
            df_filtered = df.tail(hours_needed).copy()
        else:
            # 마지막 N시간 데이터 사용
            df_filtered = df_filtered.tail(hours_needed).copy()
        
        print(f"  • 사용 데이터 기간: {df_filtered['datetime'].min()} ~ {df_filtered['datetime'].max()}")
        print(f"  • 데이터 포인트: {len(df_filtered)}시간")
        
        # 필요한 특성 컬럼만 추출
        available_features = [col for col in feature_cols if col in df_filtered.columns]
        missing_features = [col for col in feature_cols if col not in df_filtered.columns]
        
        if missing_features:
            print(f"  ⚠️ 누락된 특성: {missing_features}")
            print(f"  → 더미 데이터로 대체합니다.")
        
        # 필요한 특성 데이터 준비
        current_data = df_filtered[['datetime']].copy()
        for col in feature_cols:
            if col in available_features:
                current_data[col] = df_filtered[col].values
            else:
                # 누락된 특성은 0으로 채움
                current_data[col] = 0.0
        
    else:
        print(f"⚠️ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("더미 데이터를 생성합니다...")
        
        # 더미 데이터 생성
        end_time = current_datetime
        start_time = end_time - timedelta(hours=hours_needed-1)
        
        datetime_range = pd.date_range(start=start_time, end=end_time, freq='H')
        
        current_data = pd.DataFrame({
            'datetime': datetime_range
        })
        
        # 랜덤 특성 데이터 생성 (실제 환경에서는 센서 데이터 사용)
        for col in feature_cols:
            current_data[col] = np.random.randn(len(datetime_range)) * 0.5 + 0.5
    
    print(f"✅ 데이터 준비 완료 (shape: {current_data.shape})")
    
    return current_data


# === 예측 함수 ===
def predict_solar_generation(new_data, models_dict, sequence_length=24):
    """새로운 데이터에 대해 태양광 발전량 예측"""
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
    
    # 데이터 형식 변환 및 특성 순서 보장
    if isinstance(new_data, pd.DataFrame):
        try:
            new_data_array = new_data[feature_cols].values
        except KeyError as e:
            missing_cols = set(feature_cols) - set(new_data.columns)
            raise ValueError(f"필수 특성 컬럼이 없습니다: {missing_cols}")
    else:
        new_data_array = new_data
        if new_data_array.shape[1] != expected_features:
            raise ValueError(
                f"입력 데이터의 특성 개수({new_data_array.shape[1]})가 "
                f"예상과 다릅니다(예상: {expected_features})"
            )
    
    # 데이터가 시퀀스 길이보다 작으면 에러
    if len(new_data_array) < sequence_length:
        raise ValueError(f"데이터 길이({len(new_data_array)})가 시퀀스 길이({sequence_length})보다 작습니다.")
    
    # 결측값 처리
    new_data_imputed = new_data_array.copy()
    for col_idx in range(new_data_imputed.shape[1]):
        col_data = new_data_imputed[:, col_idx]
        nan_count = np.sum(np.isnan(col_data))
        
        if nan_count > 0:
            if np.all(np.isnan(col_data)):
                new_data_imputed[:, col_idx] = 0
            else:
                col_mean = np.nanmean(col_data)
                new_data_imputed[:, col_idx] = np.where(
                    np.isnan(col_data), 
                    col_mean, 
                    col_data
                )
    
    # 데이터 스케일링
    new_data_scaled = scaler_X.transform(new_data_imputed)
    
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


# === 현재 시점 예측 함수 ===
def predict_current_hour(current_data, models_dict, current_datetime, sequence_length=24):
    """
    현재 시점의 태양광 발전량 예측
    
    Args:
        current_data: 현재 시점까지의 데이터
        models_dict: 로드된 모델 딕셔너리
        current_datetime: 현재 시점
        sequence_length: 시퀀스 길이
    
    Returns:
        dict: 현재 시점 예측 결과
    """
    print(f"\n{'='*80}")
    print(f"⚡ 현재 시점 예측 중... ({current_datetime.strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*80}")
    
    # 마지막 sequence_length 데이터를 사용하여 현재 시점 예측
    if len(current_data) < sequence_length:
        raise ValueError(f"최소 {sequence_length}시간의 데이터가 필요합니다.")
    
    predictions = predict_solar_generation(
        new_data=current_data.tail(sequence_length + 1),
        models_dict=models_dict,
        sequence_length=sequence_length
    )
    
    # 가장 마지막 예측값이 현재 시점의 예측
    current_prediction = {
        'datetime': current_datetime,
        'lstm': float(predictions['lstm_predictions'][-1]),
        'gru': float(predictions['gru_predictions'][-1]),
        'stacked': float(predictions['stacked_predictions'][-1])
    }
    
    print(f"\n📊 현재 시점 예측 결과:")
    print(f"  • LSTM:    {current_prediction['lstm']:.2f} MWh")
    print(f"  • GRU:     {current_prediction['gru']:.2f} MWh")
    print(f"  • Stacked: {current_prediction['stacked']:.2f} MWh")
    
    return current_prediction


# === N시간 이후 예측 함수 ===
def predict_n_hours_ahead(current_data, models_dict, hours_ahead=24, sequence_length=24):
    """N시간 이후의 태양광 발전량 예측"""
    print(f"\n{'='*80}")
    print(f"🔮 {hours_ahead}시간 이후 예측 수행 중...")
    print(f"{'='*80}")
    
    required_length = sequence_length + hours_ahead
    if len(current_data) < required_length:
        raise ValueError(f"최소 {required_length}시간의 데이터가 필요합니다. (현재: {len(current_data)}시간)")
    
    # 가장 최근 데이터를 사용하여 예측
    predictions = predict_solar_generation(
        new_data=current_data.tail(required_length),
        models_dict=models_dict,
        sequence_length=sequence_length
    )
    
    # 마지막 N개 예측값 추출 (N시간 후 예측)
    future_predictions = {
        'lstm': predictions['lstm_predictions'][-hours_ahead:],
        'gru': predictions['gru_predictions'][-hours_ahead:],
        'stacked': predictions['stacked_predictions'][-hours_ahead:],
        'hours_ahead': hours_ahead
    }
    
    print(f"✅ {hours_ahead}시간 이후 예측 완료 ({len(future_predictions['stacked'])}시간)")
    
    return future_predictions


# === 다중 시간대 예측 및 저장 ===
def predict_multiple_horizons_realtime(current_data, models_dict, current_datetime, 
                                       output_dir='./prediction_results', sequence_length=24):
    """
    현재 시점 + 24H, 48H, 72H 이후의 태양광 발전량 예측 및 CSV 파일로 저장
    """
    print(f"\n{'='*80}")
    print(f"📊 실시간 다중 시간대 예측 시스템 시작")
    print(f"   기준 시각: {current_datetime.strftime('%Y년 %m월 %d일 %H시')}")
    print(f"{'='*80}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 1. 현재 시점 예측
    print(f"\n{'─'*80}")
    print(f"⚡ 현재 시점 예측")
    print(f"{'─'*80}")
    
    current_pred = predict_current_hour(current_data, models_dict, current_datetime, sequence_length)
    
    # 현재 시점 결과 저장
    current_df = pd.DataFrame([{
        '예측일시': current_datetime,
        '예측시간': '현재',
        'LSTM_예측(MWh)': current_pred['lstm'],
        'GRU_예측(MWh)': current_pred['gru'],
        'Stacked_예측(MWh)': current_pred['stacked']
    }])
    
    current_csv = os.path.join(output_dir, 'prediction_current.csv')
    current_df.to_csv(current_csv, index=False, encoding='utf-8-sig')
    print(f"  💾 현재 시점 예측 저장: {current_csv}")
    
    results['current'] = {
        'dataframe': current_df,
        'csv_path': current_csv,
        'prediction': current_pred
    }
    
    # 2. 24H, 48H, 72H 예측 수행
    for hours in [24, 48, 72]:
        print(f"\n{'─'*80}")
        print(f"🔮 {hours}시간 후 예측 수행")
        print(f"{'─'*80}")
        
        try:
            # 예측 수행
            predictions = predict_n_hours_ahead(
                current_data=current_data,
                models_dict=models_dict,
                hours_ahead=hours,
                sequence_length=sequence_length
            )
            
            # 시간 정보 생성
            time_labels = []
            datetime_labels = []
            for i in range(hours):
                time_labels.append(f'+{i+1}시간')
                datetime_labels.append(current_datetime + timedelta(hours=i+1))
            
            # DataFrame 생성
            df = pd.DataFrame({
                '예측시간': time_labels,
                '예측일시': datetime_labels,
                'LSTM_예측(MWh)': predictions['lstm'],
                'GRU_예측(MWh)': predictions['gru'],
                'Stacked_예측(MWh)': predictions['stacked']
            })
            
            # 누적 발전량 계산
            df['LSTM_누적(MWh)'] = df['LSTM_예측(MWh)'].cumsum()
            df['GRU_누적(MWh)'] = df['GRU_예측(MWh)'].cumsum()
            df['Stacked_누적(MWh)'] = df['Stacked_예측(MWh)'].cumsum()
            
            # CSV 파일로 저장
            csv_filename = f'prediction_{hours}H_{current_datetime.strftime("%Y%m%d_%H%M")}.csv'
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            print(f"  ✅ {hours}시간 예측 완료")
            print(f"  💾 파일 저장: {csv_path}")
            print(f"  📈 총 예측 발전량 (Stacked): {predictions['stacked'].sum():.2f} MWh")
            print(f"  📊 시간당 평균 (Stacked): {predictions['stacked'].mean():.2f} MWh")
            
            # 결과 저장
            results[f'{hours}H'] = {
                'dataframe': df,
                'csv_path': csv_path,
                'summary': {
                    'lstm_total': float(predictions['lstm'].sum()),
                    'gru_total': float(predictions['gru'].sum()),
                    'stacked_total': float(predictions['stacked'].sum()),
                    'lstm_mean': float(predictions['lstm'].mean()),
                    'gru_mean': float(predictions['gru'].mean()),
                    'stacked_mean': float(predictions['stacked'].mean()),
                    'lstm_max': float(predictions['lstm'].max()),
                    'gru_max': float(predictions['gru'].max()),
                    'stacked_max': float(predictions['stacked'].max()),
                }
            }
            
        except Exception as e:
            print(f"  ❌ {hours}시간 예측 실패: {e}")
            results[f'{hours}H'] = None
    
    # 3. 통합 요약 리포트 생성
    print(f"\n{'='*80}")
    print("📋 통합 예측 요약 리포트 생성")
    print(f"{'='*80}")
    
    summary_data = [{
        '예측구간': '현재',
        'LSTM_발전량(MWh)': current_pred['lstm'],
        'GRU_발전량(MWh)': current_pred['gru'],
        'Stacked_발전량(MWh)': current_pred['stacked'],
    }]
    
    for hours in [24, 48, 72]:
        if results[f'{hours}H'] is not None:
            summary = results[f'{hours}H']['summary']
            summary_data.append({
                '예측구간': f'{hours}시간',
                'LSTM_발전량(MWh)': summary['lstm_total'],
                'GRU_발전량(MWh)': summary['gru_total'],
                'Stacked_발전량(MWh)': summary['stacked_total'],
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'prediction_summary_{current_datetime.strftime("%Y%m%d_%H%M")}.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print(f"  💾 통합 요약 저장: {summary_path}")
    
    # 콘솔 출력
    print(f"\n{'='*80}")
    print("📊 최종 예측 결과 요약")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"✨ 모든 예측 결과가 '{output_dir}' 디렉토리에 저장되었습니다!")
    print(f"{'='*80}")
    print(f"\n저장된 파일:")
    print(f"  📄 {current_csv}")
    for hours in [24, 48, 72]:
        if results[f'{hours}H'] is not None:
            print(f"  📄 {results[f'{hours}H']['csv_path']}")
    print(f"  📄 {summary_path}")
    
    return results


# === 메인 실행 함수 ===
if __name__ == "__main__":
    try:
        # 현재 날짜 및 시간 자동 설정 (2025년 11월 10일)
        CURRENT_DATETIME = datetime(2025, 11, 10, datetime.now().hour)
        
        print("\n" + "="*80)
        print("🚀 실시간 태양광 발전량 예측 시스템")
        print(f"   📅 기준 시각: {CURRENT_DATETIME.strftime('%Y년 %m월 %d일 %H시')}")
        print("="*80)
        
        # 1. 저장된 모델 로드
        models = load_trained_models(model_dir='./saved_models')
        
        # 2. 현재 시점까지의 데이터 준비
        data_path = "./dataset/jeju_solar_utf8.csv"
        
        current_data = prepare_current_data(
            data_path=data_path,
            current_datetime=CURRENT_DATETIME,
            models_dict=models,
            hours_needed=96  # 72시간 예측 + 24시간 시퀀스
        )
        
        # 3. 현재 시점 + 24H/48H/72H 예측 수행 및 저장
        results = predict_multiple_horizons_realtime(
            current_data=current_data,
            models_dict=models,
            current_datetime=CURRENT_DATETIME,
            output_dir='./prediction_results',
            sequence_length=24
        )
        
        # 4. 상세 결과 미리보기
        print(f"\n{'='*80}")
        print("📋 상세 결과 미리보기")
        print(f"{'='*80}")
        
        # 현재 시점 결과
        print(f"\n⚡ 현재 시점 ({CURRENT_DATETIME.strftime('%Y-%m-%d %H:00')}):")
        print(results['current']['dataframe'].to_string(index=False))
        
        # 각 시간대 예측 결과 (처음 5개와 마지막 5개)
        for hours in [24, 48, 72]:
            if results[f'{hours}H'] is not None:
                print(f"\n🔮 {hours}시간 후 예측 (처음 5시간):")
                print(results[f'{hours}H']['dataframe'].head().to_string(index=False))
                print(f"\n🔮 {hours}시간 후 예측 (마지막 5시간):")
                print(results[f'{hours}H']['dataframe'].tail().to_string(index=False))
        
        print(f"\n{'='*80}")
        print("✅ 예측 완료!")
        print(f"{'='*80}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("먼저 모델을 학습하고 저장해야 합니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()