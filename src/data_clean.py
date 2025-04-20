import pandas as pd

df = pd.read_csv("../Data/jirasoftware.csv")
filtered = df[df.issuekey.str.startswith('JSW')]
sensitive = filtered.title.str.startswith('As ').astype(int)
filtered["is_internal"]=sensitive
filtered = filtered.fillna("")
filtered["text"] = filtered["title"]+". "+filtered["description"]
filtered.to_csv("../Data/jirasoftware_filtered.csv", index=False)