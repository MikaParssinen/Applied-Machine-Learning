import pandas as pd

df = pd.read_csv("./FIFA18_players_database/CompleteDataset.csv", low_memory=False) #
df = df.reset_index()  # make sure indexes pair with number of rows
new_df = pd.DataFrame()
for index, row in df.iterrows():
    positions = row['Preferred Positions'].split()
    if len(positions) > 1:
        df.loc[index, 'Preferred Positions'] = positions[0]
        positions = positions[1:]
        for pos in positions:
            new_row = row.copy()
            new_row['Preferred Positions'] = pos
            temp_df = new_row.to_frame().T
            new_df = pd.concat([temp_df, new_df], ignore_index=True)  # adding a row
            new_df.sort_index(inplace=True)
df = pd.concat([new_df, df], ignore_index=True)
df.drop('index', axis=1, inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.sort_values(by=['Overall'], inplace=True, ascending=False)

cols = [0, 5]
cols.extend(range(12, 46))
cols.extend([62])
df = df.iloc[:, cols]
df.to_csv('./FIFA18_players_database/clean_data.csv', index=False)