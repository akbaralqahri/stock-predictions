import yfinance as yf
import numpy as np
from random import random
import matplotlib.pyplot as plt
from scipy.stats import norm

# defining the ticker
ticker = yf.Ticker("BBNI.JK")

#obtaining historical market data
start_date = "2021-01-01"
end_date = "2021-12-31"
hist = ticker.history(start=start_date, end=end_date)
print(hist.head())

#pulling closing price data
hist = hist[['Close']]
print(hist)

#plotting price data
hist['Close'].plot(title="BBNI Stock Price", ylabel = "Closing Price", figsize=[10,6])
plt.grid()

#crete day count, price, and change lists
days = [i for i in range(1, len(hist["Close"])+1)]
price_orig = hist["Close"].tolist()
change = hist['Close'].pct_change().tolist()
change = change[1:] #remove first element

#statistics for use in model
mean = np.mean(change)
std_dev = np.std(change)
print('\nMean percent change: ' + str(round(mean*100, 2)) + '%')
print('Standard deviation: ' + str(round(std_dev*100, 2)) + '%')

#simulation number and prediction period
simulation = 10 #change for more result
days_to_sim = 1 * 130 #trading days in 1 year

#initializing figure for simulation
fig = plt.figure(figsize=[15,6])
plt.plot(days, price_orig, label='Actual Price')
plt.title('Monte Carlo Stock Price [' + str(simulation) + ' Simulations]')
plt.xlabel('Trading Days After ' + start_date)
plt.ylabel('Closing Price')
plt.xlim([200, len(days) + days_to_sim])
plt.grid()

#initialiazing lists for analysis
close_end = []
above_close = []

#for loop for number of simulation desired
for i in range(simulation):
  num_days = [days[-1]]
  close_price = [hist.iloc[-1, 0]]

  #for loop for number of days to predict
  for j in range(days_to_sim):
    num_days.append(num_days[-1] + 1)
    perc_change = norm.ppf(random(), loc = mean, scale = std_dev)
    close_price.append(close_price[-1] * (1 + perc_change))

  if close_price[-1] > close_price[0]:
    above_close.append(1)
  else:
    above_close.append(0)
  
  close_end.append(close_price[-1])
  plt.plot(num_days, close_price)

#average closing price and probability of increasing after 1 year
average_closing_price = sum(close_end) / simulation
average_perc_change = (average_closing_price - price_orig[-1]) / price_orig[-1]
probability_of_increase = sum(above_close) / simulation
print('\nPredicted closing price after ' + str(simulation) + 'simulation: $' + str(round(average_closing_price, 2)))
print('Predicted cpercent increase afrer 1 year ' + str(round(average_perc_change*100, 2)) + '%')
print('Probability of stock price increasing after 1 year: ' + str(round(probability_of_increase*100, 2)) + '%')

#displaying the monte carlo simulation
plt.show()