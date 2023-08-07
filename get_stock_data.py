import pickle

import pandas as pd
import yfinance as yf
import talib


data = yf.download(tickers="ADS.DE", interval='1d', period="5y")

target = ((data["Close"]/data["Open"] - 1) * 100).shift(periods=-1)

rsi_value = talib.RSI(data['Close'], timeperiod=11) / 100

sma_21 = talib.SMA(data["Close"], timeperiod=21) / data["Close"]
sma_50 = talib.SMA(data["Close"], timeperiod=50) / data["Close"]
sma_200 = talib.SMA(data["Close"], timeperiod=200) / data["Close"]

features = [target, rsi_value, sma_21, sma_50, sma_200]

result = pd.concat(features, axis=1)
print(result)
result.dropna(inplace=True)


with open("stock_data.pickle" , "wb") as f:
    pickle.dump(result, f)

