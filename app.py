import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(page_title="Demand Forecast", layout="wide")
st.title("📦 Demand Forecast System")

# ==========================
# Load data
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/sales.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(["id", "date"])
    df['display_name'] = df['id'] + " — " + df['name']
    return df

df = load_data()

# ==========================
# Load model & scalers
# ==========================
MODEL_FEATURES = [
    'sales', 'price', 'discounted_price', 'promotion',
    'size', 'weight', 'items_in_pack', 'sin_day', 'cos_day'
]
NUMERIC_FEATURES = ['price', 'discounted_price', 'promotion', 'size', 'weight', 'items_in_pack']

SEQ_LEN = 30

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64,1)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

@st.cache_resource
def load_model_and_scalers():
    model = LSTMModel(len(MODEL_FEATURES))
    model.load_state_dict(torch.load("model/lstm_model.pth", map_location="cpu"))
    model.eval()
    scaler_sales = joblib.load("model/scaler_sales.pkl")
    scaler_features = joblib.load("model/scaler_features.pkl")
    return model, scaler_sales, scaler_features

model, scaler_sales, scaler_features = load_model_and_scalers()

# ==========================
# Product selection
# ==========================
product_list = df[['id','name']].drop_duplicates()
product_list['display'] = product_list['id'] + " — " + product_list['name']
selected_display = st.selectbox("Выберите товар", product_list['display'])
selected_id = selected_display.split(" — ")[0]
product_df = df[df['id'] == selected_id].copy()

# ==========================
# Historical sales plot
# ==========================
st.subheader("📈 Исторические продажи")
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(product_df['date'], product_df['sales'])
ax1.set_title("Продажи по времени")
ax1.set_xlabel("Дата")
ax1.set_ylabel("Количество")
st.pyplot(fig1)

# ==========================
# Prepare features for forecast
# ==========================
st.subheader("🤖 Прогноз")
forecast_days = st.slider("Дней вперёд", 7, 60, 30)

product_df['day_of_year'] = product_df['date'].dt.dayofyear
product_df['sin_day'] = np.sin(2*np.pi*product_df['day_of_year']/365)
product_df['cos_day'] = np.cos(2*np.pi*product_df['day_of_year']/365)

# Scale numeric features
scaled = product_df.copy()
scaled[NUMERIC_FEATURES] = scaler_features.transform(scaled[NUMERIC_FEATURES])
history = scaled[MODEL_FEATURES].values[-SEQ_LEN:]
current_seq = history.copy()

last_date = product_df['date'].max()
future_preds = []

# ==========================
# Forecast loop
# ==========================
for i in range(forecast_days):
    input_tensor = torch.tensor(current_seq[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(input_tensor).item()

    dummy = np.zeros((1, len(MODEL_FEATURES)))
    dummy[0,0] = pred_scaled  # sales_scaled
    real_pred = scaler_sales.inverse_transform(dummy)[0,0]
    future_preds.append(real_pred)

    # Roll sequence forward
    new_row = current_seq[-1].copy()
    new_row[0] = pred_scaled
    # Advance day_of_year for sin/cos
    day_of_year = (int(product_df['day_of_year'].iloc[-1]) + i + 1) % 365
    new_row[-2] = np.sin(2*np.pi*day_of_year/365)
    new_row[-1] = np.cos(2*np.pi*day_of_year/365)
    current_seq = np.vstack([current_seq[1:], new_row])

future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_preds))]

# ==========================
# Forecast plot
# ==========================
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(product_df['date'], product_df['sales'], label="История")
if future_preds:
    ax2.plot(future_dates, future_preds, label="Прогноз")
ax2.set_title("Прогноз продаж")
ax2.set_xlabel("Дата")
ax2.set_ylabel("Количество")
ax2.legend()
st.pyplot(fig2)

# ==========================
# Recommended order
# ==========================
st.subheader("📦 Рекомендованный объём закупки")
safety_coef = st.slider("Коэффициент запаса", 1.0, 2.0, 1.2)
recommended = int(np.sum(future_preds) * safety_coef) if future_preds else 0
st.metric("Закупить на период", recommended)

# ==========================
# Additional stats
# ==========================
st.subheader("📊 Статистика по товару")
col1, col2, col3 = st.columns(3)
col1.metric("Средние продажи", int(product_df['sales'].mean()))
col2.metric("Макс продажи", int(product_df['sales'].max()))
col3.metric("Мин продажи", int(product_df['sales'].min()))