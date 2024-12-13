import pandas as pd


df = pd.read_csv("./Dataset/Customer Data.csv")

df.drop(['TENURE', 'CUST_ID'], axis=1, inplace=True)
df = df.astype({'PURCHASES_TRX': 'float', 'CASH_ADVANCE_TRX': 'float'})
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df.to_csv("./Dataset/Clean.csv", index=False)
