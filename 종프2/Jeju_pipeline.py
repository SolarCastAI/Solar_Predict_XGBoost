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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from xgboost import plot_importance

# 날짜, 기온, 강수량, 일조, 일사량 -> 태양광 발전량 

data_path = "C:/Users/rlask/종프2/dataset/jeju_solar_utf8.csv"
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-whitegrid')

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


class SolarDataset(Dataset):
    def __init__(self, features, targets, seq_len=24):
        self.X, self.y = [], []
        
        # 시퀀스 데이터 생성
        for i in range(len(features) - seq_len):
            self.X.append(features[i:i+seq_len])
            self.y.append(targets[i+seq_len])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    



class CNN_LSTM(nn.Module):
    def __init__(self, input_dim=4, seq_len=24, hidden_dim=64, num_layers=1):
        super(CNN_LSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        # LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # FC layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)   # (batch, features, seq_len)
        x = self.conv1(x)        
        x = self.relu(x)
        x = self.pool(x)         # (batch, 32, seq_len//2)

        x = x.permute(0, 2, 1)   # (batch, seq_len//2, 32)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)  
        out = h_n[-1]               # 마지막 hidden state 사용

        # FC layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()
    
    def train_model(self, train_loader, val_loader=None, epochs=50, lr=0.001):
        """
        모델 학습 함수 - 매 에포크마다 진행상황 출력
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        train_losses = []
        val_losses = []
        
        print(f"총 {epochs} 에포크 학습을 시작합니다...")
        start_time = time.time()
        
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
                scheduler.step(avg_val_loss)
                
                # 매 에포크마다 출력
                epoch_time = time.time() - epoch_start_time
                elapsed_time = time.time() - start_time
                remaining_epochs = epochs - epoch - 1
                estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs
                
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"시간: {epoch_time:.1f}s | "
                      f"예상 남은 시간: {estimated_remaining/60:.1f}분")
            else:
                # validation loader가 없는 경우
                epoch_time = time.time() - epoch_start_time
                elapsed_time = time.time() - start_time
                remaining_epochs = epochs - epoch - 1
                estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs
                
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {avg_train_loss:.4f} | "
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


def xgb_model(X_train, y_train, X_val, y_val, plotting=False):
    # XGBoost 모델 정의
    xgb_regressor = xgb.XGBRegressor(
        gamma=1, 
        n_estimators=200, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42,
        early_stopping_rounds=50 
    )
    
    # 모델 학습
    xgb_regressor.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False # 학습 과정을 출력하지 않음
    )
    
    # 검증 데이터에 대한 예측
    pred_val = xgb_regressor.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    
    # 결과 시각화
    if plotting:
        plt.figure(figsize=(15, 6))
        sns.lineplot(x=range(len(y_val)), y=y_val.values, color="grey", alpha=.4, label="Actual")
        sns.lineplot(x=range(len(y_val)), y=pred_val, color="red", label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("DC Power")
        plt.title(f"XGBoost Validation - MAE: {round(mae, 3)}")
        plt.legend()
        plt.show()
        
    return mae, xgb_regressor


def create_sequences_and_split(features, targets, seq_len=24, test_size=0.2, val_size=0.1):

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    

    dataset = SolarDataset(features_scaled, targets_scaled, seq_len)
    

    dataset_size = len(dataset)
    test_split = int(dataset_size * test_size)
    val_split = int(dataset_size * val_size)
    train_split = dataset_size - test_split - val_split
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_split, val_split, test_split]
    )
    
    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler


if __name__ == "__main__":
    try:
        print("데이터 로딩 중...")
        data_df = pd.read_csv(data_path)

        features = data_df[["기온", "강수량(mm)", "일조(hr)", "일사량"]].values
        targets = data_df["태양광 발전량(MWh)"].values
        
        print(f"데이터 크기: {len(data_df)} 행")
        print("데이터 전처리 중...")

        seq_len = 24
        train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = create_sequences_and_split(
            features, targets, seq_len=seq_len
        )
        

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # 모델 생성
        model = CNN_LSTM(input_dim=4, seq_len=seq_len, hidden_dim=64, num_layers=2)
        print("모델 생성 완료")
        
        # 모델 학습
        print("\n" + "="*60)
        print("모델 학습 시작...")
        print("="*60)
        train_losses, val_losses = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,  # 필요하면 여기서 에포크 수를 줄일 수 있습니다
            lr=0.001
        )
        
        # 테스트 예측
        print("\n테스트 예측 중...")
        predictions, actuals = model.predict(test_loader)
        
        # 원본 스케일로 변환
        predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 성능 평가
        mae = mean_absolute_error(actuals_original, predictions_original)
        rmse = calculate_rmse(actuals_original, predictions_original)
        mape = calculate_mape(actuals_original, predictions_original)
        
        print(f"\n=== 모델 성능 평가 ===")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}%")
        
        # 결과 시각화
        plt.figure(figsize=(15, 10))
        
        # 손실 곡선
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 예측 vs 실제값 (전체)
        plt.subplot(2, 2, 2)
        plt.scatter(actuals_original, predictions_original, alpha=0.5)
        plt.plot([actuals_original.min(), actuals_original.max()], 
                 [actuals_original.min(), actuals_original.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual')
        plt.grid(True)
        
        # 시계열 예측 결과 (일부)
        plt.subplot(2, 2, 3)
        sample_size = min(200, len(actuals_original))
        plt.plot(actuals_original[:sample_size], label='Actual', alpha=0.8)
        plt.plot(predictions_original[:sample_size], label='Predicted', alpha=0.8)
        plt.title('Time Series Prediction (Sample)')
        plt.xlabel('Time')
        plt.ylabel('Solar Power Generation (MWh)')
        plt.legend()
        plt.grid(True)
        
        # 잔차 분석
        plt.subplot(2, 2, 4)
        residuals = actuals_original - predictions_original
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.title('Residuals Distribution')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("학습 완료!")
        
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()