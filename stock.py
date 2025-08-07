import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import ta  # Technical analysis library (alternative to talib)

# ========================
# SETUP & DATA COLLECTION
# ========================

# Configure plotting
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Download data (5 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)
ticker = 'AAPL'

print(f"Downloading {ticker} data...")
data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

# Flatten MultiIndex columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ========================
# FEATURE ENGINEERING
# ========================

def add_features(df):
    """Add technical indicators and lagged features"""
    # Basic price features
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(21).std()
    
    # Technical indicators (using ta library instead of talib)
    # Ensure we're working with pandas Series and handle any data issues
    close_series = df['Close'].copy()
    
    # Simple RSI calculation as fallback
    try:
        df['RSI_14'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    except Exception as e:
        print(f"RSI calculation failed: {e}")
        # Manual RSI calculation
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Simple EMA calculation
    try:
        df['EMA_20'] = ta.trend.EMAIndicator(close_series, window=20).ema_indicator()
    except Exception as e:
        print(f"EMA calculation failed: {e}")
        df['EMA_20'] = close_series.ewm(span=20).mean()
    
    # Simple SMA calculations
    try:
        df['SMA_50'] = ta.trend.SMAIndicator(close_series, window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close_series, window=200).sma_indicator()
    except Exception as e:
        print(f"SMA calculation failed: {e}")
        df['SMA_50'] = close_series.rolling(window=50).mean()
        df['SMA_200'] = close_series.rolling(window=200).mean()
    
    # Lagged features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    # Volume features
    df['Volume_MA'] = df['Volume'].rolling(10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df.dropna()

data = add_features(data)
features = ['Close', 'Volume', 'RSI_14', 'EMA_20', 'SMA_50', 'Volatility']

# ========================
# DATA PREPROCESSING
# ========================

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])

# Sequence creation
def create_sequences(data, seq_length, target_col=0):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_col])  # Predict Close price
    return np.array(X), np.array(y)

seq_length = 60  # Optimal lookback period
X, y = create_sequences(scaled_data, seq_length)

# Train-test split (80-20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ========================
# LSTM MODEL ARCHITECTURE
# ========================

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# ========================
# MODEL TRAINING
# ========================

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=185,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ========================
# EVALUATION & PREDICTIONS
# ========================

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(
    np.concatenate((predictions, np.zeros((len(predictions), len(features)-1))), axis=1)
)[:, 0]

# Prepare results DataFrame
test_dates = data.index[split+seq_length+1:split+seq_length+1+len(predictions)]
results = pd.DataFrame({
    'Actual': data['Close'].values[split+seq_length+1:split+seq_length+1+len(predictions)],
    'Predicted': predictions
}, index=test_dates)

# Smooth predictions
results['Smoothed'] = results['Predicted'].ewm(span=3).mean()

# Calculate RMSE
rmse = np.sqrt(np.mean((results['Actual'] - results['Predicted'])**2))
print(f"\nFinal RMSE: {rmse:.2f}")

# ========================
# VISUALIZATION
# ========================

plt.figure(figsize=(16,8))
plt.plot(data.index[:split+seq_length+1], data['Close'][:split+seq_length+1], label='Training Data')
plt.plot(results.index, results['Actual'], label='Actual Price', alpha=0.7)
plt.plot(results.index, results['Predicted'], label='Raw Predictions', alpha=0.5)
plt.plot(results.index, results['Smoothed'], label='Smoothed Predictions', linewidth=2)
plt.title(f'{ticker} Stock Price Prediction (RMSE: {rmse:.2f})')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=300, bbox_inches='tight')
print("Prediction plot saved as 'stock_prediction.png'")

# Plot training history
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Progress')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training history plot saved as 'training_history.png'")

print(f"\n✅ Stock prediction model completed successfully!")
print(f"📊 Final RMSE: {rmse:.2f}")
print(f"📈 Model trained on {len(X_train)} samples, tested on {len(X_test)} samples")
print(f"🎯 Predictions saved to 'stock_prediction.png'")
print(f"📉 Training history saved to 'training_history.png'")