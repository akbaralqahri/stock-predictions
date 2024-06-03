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
ax.plot(hist['Close'], label = 'historical')
ax.plot(actual['Close'], label = 'actual')
ax.plot(predic['Close'], linestyle='--', label = 'prediction')

ax.set_xlabel('Date')
ax.set_ylabel('Close')

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

# estimasi MOdel CRR sigma
logS= np.log(np.array([hist.Close[i] for i in range(0,int(len(hist)/2))]))
print('varoansi log saham($sigma^2t$)=', np.std(logS))
sig = np.sqrt(np.std(logS)/1)# volatilitas harga saham

import math
# Estimasi nilai sigma dari log(S)
hist['logs'] = [np.log(k) for k in hist['Close']]
logs = np.array(hist['logs'])[0:int(len(hist)/2)+1] 
varLogs=np.std(logs)**2 #\sigma^2t
sigEst= np.sqrt(varLogs/1)
print(len(hist['logs'][int(len(hist)/2)+1:-1]))

# Membuat pohon Binomial Saham
So = hist.Close[int(len(hist)/2)]
sig = sigEst
r = 0.055 # suku bunagn bank Indonesia
N = 100#251 # harian dalam satu tahun
dt = 1/N
u=np.exp(sig*np.sqrt(dt))
d=1/u
p=(np.exp(-r*dt)-d)/(u-d)

fig,ax = plt.subplots(2,figsize=(8,4))
price = np.zeros((N+1,N+1), float)
mean_si = np.zeros((N+1), float)
mean2_si = np.zeros((N+1), float)
mean3_si = np.zeros((N+1), float)
mean_si[0] = So
mean2_si[0] = So
mean3_si[0] = So
price[0,N] = So
ax[0].scatter(0, price[0,N])
ax[1].plot(range(len(hist.Close[int(len(hist)/2)+1:-1])), hist.Close[int(len(hist)/2)+1:-1], label='harga aktual BBCA')
for i in range(1,N//24): # time step 
  for j in reversed(range(i+1)): # jumlah kenaikan
    price[j,i] = So*u**(j)*d**(i-j)
  ax[0].scatter([i for k in range(len(price[0:i+1,i]))], price[0:i+1,i])
  mean_si[i] = np.mean(price[0:i+1,i])
  prob = np.array([math.comb(i+1, i+1-k)*p**(i+1-k)*(1-p)**k for k in range(i+1)])
  # print(prob.shape,price[0:i+1,i].shape )
  mean2_si[i] = np.sum(price[0:i+1,i]*prob) 
  mean3_si[i] = So*u**(i*p)*d**(i-i*p)
    # print(price[j,i])
    # print(j)
ax[0].set_title("Binomial Tree")
ax[1].plot(mean_si, label='mean S(i)')
ax[1].plot(mean2_si, label='mean2 S(i)')
ax[1].plot(mean3_si, label='mean3 S(i)')
plt.legend()

a = np.array([i for i in range(3)])
b = np.array([i for i in reversed(range(3))])
a*b
print(a.shape, b.shape)

plt.plot(price)
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

[plt.plot(k) for k in smt]
plt.show()
## Average path of stock prices
smean=np.mean(np.array(smt), axis=0)
plt.plot(smean)

# Binomial Tree
so=10
T=1
N=10
dt=T/N
r = 0.06 
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