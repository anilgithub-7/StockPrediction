# StockPrediction
Stock Prediction with LSTM and RMSE below 8 for Apple
# Stock Price Prediction with LSTM

A machine learning project that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. This project analyzes Apple Inc. (AAPL) stock data and provides both basic and optimized prediction models.

## Features

- **Data Collection**: Downloads 5 years of historical stock data using Yahoo Finance
- **Feature Engineering**: Implements technical indicators including:
  - RSI (Relative Strength Index)
  - EMA (Exponential Moving Average)
  - SMA (Simple Moving Average)
  - Volatility measures
  - Volume analysis
- **LSTM Model**: Deep learning model for time series prediction
- **Hyperparameter Optimization**: Automated model tuning for better performance
- **Visualization**: Comprehensive charts and analysis plots

## Requirements

- Python 3.8+
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- yfinance
- ta (Technical Analysis library)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd stockpredict
```

2. Create a virtual environment:
```bash
python -m venv env
```

3. Activate the virtual environment:
```bash
# Windows
env\Scripts\activate

# macOS/Linux
source env/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python stock.py
```

The script will:
1. Download AAPL stock data
2. Preprocess and engineer features
3. Train an LSTM model
4. Generate predictions
5. Create visualization plots

## Output Files

- `stock_prediction.png`: Basic model predictions
- `training_history.png`: Training loss visualization
- `optimized_stock_prediction.png`: Optimized model predictions
- `optimized_training_history.png`: Optimized training history
- `AAPL_enhanced_analysis.png`: Enhanced analysis with technical indicators

## Model Architecture

The LSTM model includes:
- LSTM layers for sequence learning
- Dropout layers for regularization
- Dense layers for final predictions
- Early stopping to prevent overfitting

## Technical Indicators

- **RSI**: Measures momentum and identifies overbought/oversold conditions
- **EMA/SMA**: Trend-following indicators
- **Volatility**: Risk assessment metric
- **Volume Analysis**: Market participation indicators

## Disclaimer

This project is for educational purposes only. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting with financial advisors before making investment decisions.

## License

MIT License - feel free to use this code for educational and research purposes.
