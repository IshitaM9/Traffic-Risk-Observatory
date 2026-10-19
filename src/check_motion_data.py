
from pathlib import Path
import pandas as pd

path = Path("outputs/motion_data.csv")
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())
print("Rows:", len(df))
print("Unique tracked IDs:", df["track_id"].nunique())
print("Detected classes:")
print(df["class"].value_counts().to_string())

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))
