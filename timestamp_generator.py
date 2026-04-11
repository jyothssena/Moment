import pandas as pd
import random
from datetime import datetime, timezone

df = pd.read_csv("/Users/tanmayishurpali/mar24/Moment/interpretations_train.csv")

def random_timestamp_2025():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    random_seconds = random.randint(0, int((end - start).total_seconds()))
    dt = start + pd.Timedelta(seconds=random_seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

df["timestamp"] = [random_timestamp_2025() for _ in range(len(df))]

df.to_csv("/Users/tanmayishurpali/mar24/Moment/interpretations_train_with_timestamps.csv", index=False)