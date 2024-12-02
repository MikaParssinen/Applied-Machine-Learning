import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def clean_plus_minus(value):
    if isinstance(value, str):
        return re.sub(r'[+-].*', '',value)
    return value

pd.options.mode.copy_on_write = True  # Enable copy on write in pandas
df = pd.read_csv("../FIFA18_players_database/CompleteDataset.csv", low_memory=False)  # Open file to clean
df = df.reset_index()  # Make sure indexes pair with number of rows

cols = list(range(14, 48))
cols.append(64)
df = df.iloc[:, cols]  # Extract only columns of interest
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the rows of the
df = df[~df["Preferred Positions"].str.contains("GK")]  # Remove all goalkeeper rows
cols_to_drop = [col for col in df.columns if "GK" in col]  # Find all GK attributes
df.drop(cols_to_drop, axis=1, inplace=True)  # Remove columns
df = df.map(clean_plus_minus)  # Remove +, -.

# Split rows with multiple labels into multiple rows
df['Preferred Positions'] = df['Preferred Positions'].str.strip()  # Remove whitespaces
df["Preferred Positions"] = df["Preferred Positions"].str.split(" ")
df = df.explode("Preferred Positions")

df.to_csv("./FIFA18_players_database/clean_data.csv", index=False)

# Normalize with all numeric values with MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

df.to_csv("./FIFA18_players_database/clean_data_normalized.csv", index=False)