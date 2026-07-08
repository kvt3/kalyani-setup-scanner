

import yfinance as yf

# Choose your tracking ticker
ticker = yf.Ticker("AAPL")
#print(ticker.info)
# Fetch the earnings/revenue historical matrix
#earning_history = ticker.earnings_dates
#revenue_history = ticker.get_revenue_estimate
print(ticker.get_earnings_dates())

#print(earning_history) 

actual = ticker.quarterly_income_stmt.loc["Total Revenue"]

# Revenue estimates
estimate = ticker.revenue_estimate

print("Actual Revenue:",actual)
#print(actual)

print("\nRevenue Estimates:",estimate)
#print(estimate)


