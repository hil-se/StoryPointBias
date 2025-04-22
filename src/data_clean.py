import pandas as pd

df = pd.read_csv("../Data/jirasoftware.csv")
df = df[df.issuekey.str.startswith('JSW')]
sensitive = df.title.str.startswith('As ').astype(int)
df["is_internal"]=sensitive
df = df.fillna("")
df["text"] = df["title"]+"\r"+df["description"]
df.to_csv("../Data/jirasoftware_filtered.csv", index=False)