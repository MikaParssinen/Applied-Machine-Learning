import pandas as pd

df = pd.read_csv("./FIFA18_players_database/CompleteDataset.csv", low_memory=False) # Open file to clean
df = df.reset_index()   # Make sure indexes pair with number of rows
new_df = pd.DataFrame() # Create dataframe for new rows that need to be added to the original dataframe
for index, row in df.iterrows():
    positions = row['Preferred Positions'].split()          # Get a list of preferred positions
    if len(positions) > 1:                                  # If the list contains more than one item
        df.loc[index, 'Preferred Positions'] = positions[0] # Changed preferred position to only contain 1 position
        positions = positions[1:]                           # Remove first position from list
        for pos in positions:                               # For each position left in the list:
            new_row = row.copy()                            # Make a new row and copy values from row
            new_row['Preferred Positions'] = pos            # Change preferred position to current position
            temp_df = new_row.to_frame().T                  # Make new row to a DataFrame by taking the
                                                            # series new_row and transposing it
            new_df = pd.concat([temp_df, new_df], ignore_index=True)  # Adding a row to new dataframe
            new_df.sort_index(inplace=True)
df = pd.concat([new_df, df], ignore_index=True) # Add all new rows created by concattenating original
                                                     # dataframe with new dataframe
df.drop('index', axis=1, inplace=True)        # Remove index and Unnamed: 0 columns
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(inplace=True)
df.sort_values(by=['Overall'], inplace=True, ascending=False) # Sort the dataframes by column Overall in
                                                              # descending order

cols = [0, 5]
cols.extend(range(12, 46))
cols.extend([62])
df = df.iloc[:, cols]
df.to_csv('./FIFA18_players_database/clean_data.csv', index=False)