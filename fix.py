import pandas as pd # type: ignore

df = pd.read_csv("interpretations_test.csv")

# Replace newlines inside text with a space (or \n literal if you prefer)
df["interpretation"] = df["interpretation"].str.replace("\n", " ", regex=False)

df.to_csv("test_interpretations_clean.csv", index=False)