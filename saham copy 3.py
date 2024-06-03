# !pip install yfinance
# !pip install pandas-datareader
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr

ybbni = yf.Ticker("BBNI.JK")
ybbca = yf.Ticker("BMRI.JK")

# get historical market data
hist = ybbni.history(start="2021-01-01", end="2021-12-31")
actual = ybbni.history(start="2022-01-01", end="2022-06-30")
predic = ybbca.history(start="2022-01-01", end="2022-06-30")

fig,ax = plt.subplots(1, figsize=(8,5))
ax.plot(hist['Close'], color = 'black', label = 'historical')
ax.plot(actual['Close'], color = 'red', label = 'actual')
ax.plot(predic['Close'], color = 'green', linestyle='--', label = 'prediction')

ax.set_xlabel('Date')
ax.set_ylabel('Close')

plt.legend(frameon=False)  
plt.show()

# estimasi parameter metode empiris 
hist['return'] = [hist.Close[i]/hist.Close[i-1] for i in range(1,len(hist))]+[0]
ret = np.array(hist['return'][0:int(len(hist)/2)])
fig,ax = plt.subplots(1,figsize=(10,6))
ax.plot(ret)
plt.show()

print('mean, std up', np.mean(ret[ret>1]), np.std(ret[ret>1]))
print('mean, std down',np.mean(ret[ret<=1]), np.std(ret[ret<=1]))
plt.hist(ret, bins=10)
up= np.mean(ret[ret>1]) # estimasi nilai u empiris
dwn = np.mean(ret[ret<=1])# estimasi nilai d emspiris
p_empi = len(ret[ret>1])/len(ret)
avret = np.mean(ret)
print('u,d,p=', up,dwn,p_empi)
# sigm=(ret-avret)**2
plt.show()

# MOnte carlo simulation per time steps (Bernouli event for every time step)
so=10
T=1
N=10
dt=T/N
r = 0.055 # suku bunagn bank Indonesia
sig = 0.15
u=np.exp(sig*np.sqrt(dt))
d=1/u
p=(np.exp(-r*dt)-d)/(u-d)
M=20
smt = [[]]
for k in range(M):
  s=so
  sv=[s]
  for i in range(N):
    x=np.random.binomial(N,p)
    s=s*u**(x)*d**(1-x)
    sv=sv+[s]
  smt =smt+[sv] 

smt=[k for k in smt[1:][:]]  