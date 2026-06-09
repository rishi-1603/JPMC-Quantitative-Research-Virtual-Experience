import pandas as pd
import numpy as np

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

fico_col = None

for col in df.columns:
    if "fico" in col.lower():
        fico_col = col
        break

if fico_col is None:
    raise ValueError("FICO score column not found")

NUM_BUCKETS = 10

df["Rating"] = pd.qcut(
    df[fico_col],
    q=NUM_BUCKETS,
    labels=False,
    duplicates="drop"
)

df["Rating"] = (
    df["Rating"].max() - df["Rating"] + 1
)

bucket_bounds = pd.qcut(
    df[fico_col],
    q=NUM_BUCKETS,
    duplicates="drop"
)

print("Bucket Boundaries:")
print(bucket_bounds.cat.categories)

print("\nSample Ratings:")
print(df[[fico_col, "Rating"]].head())

def assign_rating(fico_score):
    intervals = bucket_bounds.cat.categories

    for i, interval in enumerate(intervals):
        if fico_score in interval:
            return len(intervals) - i

    if fico_score < intervals[0].left:
        return len(intervals)

    return 1

sample_score = 720
print(f"\nRating for FICO {sample_score}:",
      assign_rating(sample_score))