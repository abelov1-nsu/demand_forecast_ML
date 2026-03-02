import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

torch.manual_seed(42)
np.random.seed(42)

# ==========================
# 1. Загрузка и подготовка данных
# ==========================
df = pd.read_csv("data/sales.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(["id", "date"])

# ==========================
# 2. Сезонные признаки
# ==========================
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2*np.pi*df['day_of_year']/365)
df['cos_day'] = np.cos(2*np.pi*df['day_of_year']/365)

# ==========================
# 3. Разделение на train/test
# ==========================
split_date = df['date'].quantile(0.8)
train_df = df[df['date'] <= split_date].copy()
test_df  = df[df['date'] > split_date].copy()

# ==========================
# 4. Масштабирование
# ==========================
NUMERIC_FEATURES = ['price', 'discounted_price', 'promotion', 'size', 'weight', 'items_in_pack']

scaler_features = StandardScaler()
train_df[NUMERIC_FEATURES] = scaler_features.fit_transform(train_df[NUMERIC_FEATURES])
test_df[NUMERIC_FEATURES]  = scaler_features.transform(test_df[NUMERIC_FEATURES])


# Отдельное масштирование для sales, так как это целевая переменная
scaler_sales = StandardScaler()
train_df['sales_scaled'] = scaler_sales.fit_transform(train_df[['sales']])
test_df['sales_scaled']  = scaler_sales.transform(test_df[['sales']])

MODEL_FEATURES = ['sales_scaled'] + NUMERIC_FEATURES + ['sin_day', 'cos_day']

# ==========================
# 5. Данные для обучения
# ==========================
SEQ_LEN = 120

class SalesDataset(Dataset):
    def __init__(self, data):
        self.X = []
        self.y = []

        for product_id in data['id'].unique():
            product_data = data[data['id'] == product_id].sort_values("date")
            values = product_data[MODEL_FEATURES].values

            for i in range(SEQ_LEN, len(values)):
                self.X.append(values[i-SEQ_LEN:i])
                self.y.append(values[i][0])  # sales_scaled

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_test_dataset(train_df, test_df):
    X_test = []
    y_test = []

    for product_id in test_df['id'].unique():

        train_part = train_df[train_df['id'] == product_id].sort_values("date")
        test_part  = test_df[test_df['id'] == product_id].sort_values("date")

        if len(train_part) < SEQ_LEN:
            continue  # если мало истории — пропускаем

        # берём последние SEQ_LEN дней train
        history = train_part.tail(SEQ_LEN)

        # объединяем с test
        combined = pd.concat([history, test_part])

        values = combined[MODEL_FEATURES].values

        # формируем последовательности
        for i in range(SEQ_LEN, len(values)):
            X_test.append(values[i-SEQ_LEN:i])
            y_test.append(values[i][0])  # sales_scaled

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    return torch.utils.data.TensorDataset(X_test, y_test)

train_dataset = SalesDataset(train_df)
test_dataset = create_test_dataset(train_df, test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)


# ==========================
# 6. Создание модели
# ==========================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

model = LSTMModel(len(MODEL_FEATURES))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================
# 7. Обучение модели
# ==========================
EPOCHS = 10

print("\n========== Training ==========")
print(f"Train samples: {len(train_dataset)}")
print(f"Test samples:  {len(test_dataset)}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Features: {MODEL_FEATURES}")
print("==============================\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()

    if len(test_loader) > 0:
        val_loss /= len(test_loader)
    else:
        val_loss = np.nan

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"| Train Loss: {train_loss:.4f} "
          f"| Val Loss: {val_loss:.4f}")

# ==========================
# 8. Оценка модели
# ==========================
model.eval()
all_preds, all_actual = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        all_preds.extend(preds.numpy())
        all_actual.extend(y_batch.numpy())

if len(all_preds) > 0:
    all_preds_real = scaler_sales.inverse_transform(
        np.array(all_preds).reshape(-1,1)
    )[:,0]

    all_actual_real = scaler_sales.inverse_transform(
        np.array(all_actual).reshape(-1,1)
    )[:,0]

    mae = mean_absolute_error(all_actual_real, all_preds_real)
    rmse = np.sqrt(np.mean((all_actual_real - all_preds_real)**2))
    mape = np.mean(
        np.abs((all_actual_real - all_preds_real) /
               np.maximum(all_actual_real, 1))
    ) * 100

    baseline_preds = []
    for X_batch, _ in test_loader:
        last_sales_scaled = X_batch[:, -1, 0].numpy()
        baseline_preds.extend(last_sales_scaled)

    baseline_real = scaler_sales.inverse_transform(
        np.array(baseline_preds).reshape(-1,1)
    )[:,0]

    baseline_mae = mean_absolute_error(all_actual_real, baseline_real)

else:
    mae = rmse = mape = baseline_mae = np.nan

print("\n========== Evaluation ==========")
print(f"MAE   : {mae:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"MAPE  : {mape:.2f}%")
print(f"Baseline MAE (last value): {baseline_mae:.4f}")
print("================================\n")

if mae < baseline_mae:
    print("Model outperforms baseline.")
else:
    print("Model does NOT outperform baseline.")


# ==========================
# 9. Save model & scalers
# ==========================
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/lstm_model.pth")
joblib.dump(scaler_sales, "model/scaler_sales.pkl")
joblib.dump(scaler_features, "model/scaler_features.pkl")
print("Model and scalers saved in 'model/'")

