import pandas as pd
import numpy as np


def loadData(dataName="jirasoftware_filtered"):
    path = "../Data/"
    df = pd.read_csv(path+dataName+".csv")
    return df

data = loadData()
df = data[data["split_mark"] == "test"]
n = len(df)
pair = []
for i in range(n):
    for j in range(i+1,n):
        if df["storypoint"].iloc[i] > df["storypoint"].iloc[j]:
            y = 1
        elif df["storypoint"].iloc[i] < df["storypoint"].iloc[j]:
            y = -1
        else:
            y = 0
        pair.append({"A": df["text"].iloc[i], "B": df["text"].iloc[j], "Y": y})
df_pair = pd.DataFrame(pair)
df_shuffled = df_pair.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_csv("../Data/test_pairs.csv", index=False)