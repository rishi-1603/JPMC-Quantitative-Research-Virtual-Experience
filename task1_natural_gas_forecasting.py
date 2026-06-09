import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Nat_Gas.csv')

df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values('Dates')

plt.figure(figsize=(10,5))
plt.plot(df['Dates'], df['Prices'], marker='o')
plt.title('Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Numerical time index
df['Time'] = (df['Dates'] - df['Dates'].min()).dt.days

interp_func = interp1d(
    df['Time'],
    df['Prices'],
    kind='linear',
    fill_value='extrapolate'
)

# Trend model
X = df[['Time']]
y = df['Prices']

trend_model = LinearRegression()
trend_model.fit(X, y)

df['Month'] = df['Dates'].dt.month
monthly_avg = df.groupby('Month')['Prices'].mean()

def estimate_price(date_str):
    target_date = pd.to_datetime(date_str)
    min_date = df['Dates'].min()
    max_date = df['Dates'].max()

    days = (target_date - min_date).days

    if target_date <= max_date:
        return float(interp_func(days))

    trend_price = trend_model.predict([[days]])[0]

    month = target_date.month
    seasonal_adjustment = (
        monthly_avg[month] - monthly_avg.mean()
    )

    return float(trend_price + seasonal_adjustment)

print("Estimated Price:",
      round(estimate_price('2025-06-30'), 2))