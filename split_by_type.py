import pandas as pd

df = pd.read_csv("data/merged_data.csv")

has_type = df[df["type"].notna()]
missing_type = df[df["type"].isna()]

has_type.to_csv("data/with_type.csv", index=False)
missing_type.to_csv("data/without_type.csv", index=False)

print(f"Saved {len(has_type)} rows to data/with_type.csv")
print(f"Saved {len(missing_type)} rows to data/without_type.csv")
