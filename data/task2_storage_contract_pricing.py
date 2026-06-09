import pandas as pd
from scipy.interpolate import interp1d

df = pd.read_csv("Nat_Gas.csv")

df["Dates"] = pd.to_datetime(df["Dates"])
df = df.sort_values("Dates")

df["Days"] = (df["Dates"] - df["Dates"].min()).dt.days

price_curve = interp1d(
    df["Days"],
    df["Prices"],
    kind="linear",
    fill_value="extrapolate"
)

def get_price(date):
    date = pd.to_datetime(date)
    days = (date - df["Dates"].min()).days
    return float(price_curve(days))

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    injection_rate,
    withdrawal_rate,
    max_volume,
    storage_cost_per_month
):

    volume = 0
    value = 0

    for date in injection_dates:
        injected = min(injection_rate, max_volume - volume)
        price = get_price(date)

        value -= injected * price
        volume += injected

    for date in withdrawal_dates:
        withdrawn = min(withdrawal_rate, volume)
        price = get_price(date)

        value += withdrawn * price
        volume -= withdrawn

    start = min(pd.to_datetime(injection_dates))
    end = max(pd.to_datetime(withdrawal_dates))

    storage_months = ((end - start).days) / 30

    storage_cost = (
        storage_months
        * storage_cost_per_month
        * max_volume
    )

    value -= storage_cost

    return value

contract_value = price_storage_contract(
    injection_dates=["2024-06-01", "2024-07-01"],
    withdrawal_dates=["2024-12-01", "2025-01-01"],
    injection_rate=1000,
    withdrawal_rate=1000,
    max_volume=2000,
    storage_cost_per_month=0.05
)

print("Contract Value:", round(contract_value, 2))