# !pip install yfinance
# !pip install pandas-datareader
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr

ybbca = yf.Ticker("BBNI.JK")
ybbni = yf.Ticker("BBCA.JK")

# get historical market data
hist = ybbca.history(start="2020-01-01", end="2020-12-31")
actual = ybbca.history(start="2021-01-01", end="2021-06-30")
simulation = ybbni.history(start="2021-01-01", end="2021-06-30")

fig,ax = plt.subplots(1, figsize=(8,5))
ax.plot(hist['Close'])
ax.plot(actual['Close'])
ax.plot(simulation['Close'], linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Close')

plt.show()