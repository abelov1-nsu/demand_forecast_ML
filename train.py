import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

# ==========================
# 1. Load data
# ==========================
df = pd.read_csv("data/sales.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(["id", "date"])

# ==========================
# 2. Seasonal features
# ==========================
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_day'] = np.sin(2*np.pi*df['day_of_year']/365)
df['cos_day'] = np.cos(2*np.pi*df['day_of_year']/365)

# ==========================
# 3. Train/test split
# ==========================
split_date = df['date'].quantile(0.8)
train_df = df[df['date'] <= split_date].copy()
test_df  = df[df['date'] > split_date].copy()

# ==========================
# 4. Scaling
# ==========================
NUMERIC_FEATURES = ['price', 'discounted_price', 'promotion', 'size', 'weight', 'items_in_pack']

scaler_features = StandardScaler()
train_df[NUMERIC_FEATURES] = scaler_features.fit_transform(train_df[NUMERIC_FEATURES])
test_df[NUMERIC_FEATURES]  = scaler_features.transform(test_df[NUMERIC_FEATURES])

# Leave sin_day and cos_day unscaled

# Scale sales separately
scaler_sales = StandardScaler()
train_df['sales_scaled'] = scaler_sales.fit_transform(train_df[['sales']])
test_df['sales_scaled']  = scaler_sales.transform(test_df[['sales']])

MODEL_FEATURES = ['sales_scaled'] + NUMERIC_FEATURES + ['sin_day', 'cos_day']

# ==========================
# 5. Dataset
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

train_dataset = SalesDataset(train_df)
test_dataset  = SalesDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)

# ==========================
# 6. Model
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
# 7. Training
# ==========================
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# ==========================
# 8. Evaluation
# ==========================
model.eval()
all_preds, all_actual = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        all_preds.extend(preds.numpy())
        all_actual.extend(y_batch.numpy())

if len(all_preds) > 0:
    all_preds_real = scaler_sales.inverse_transform(np.array(all_preds).reshape(-1,1))[:,0]
    all_actual_real = scaler_sales.inverse_transform(np.array(all_actual).reshape(-1,1))[:,0]
    model_mae = mean_absolute_error(all_actual_real, all_preds_real)
else:
    model_mae = np.nan

print(f"\nModel MAE: {model_mae:.4f}")

# ==========================
# 9. Save model & scalers
# ==========================
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/lstm_model.pth")
joblib.dump(scaler_sales, "model/scaler_sales.pkl")
joblib.dump(scaler_features, "model/scaler_features.pkl")
print("Model and scalers saved in 'model/'")