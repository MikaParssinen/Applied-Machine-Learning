import pandas as pd
import re

def clean_plus_minus(value):
    if isinstance(value, str):
        return re.sub(r'[+-].*', '',value)
    return value

def split_data():
    pd.options.mode.copy_on_write = True  # Enable copy on write in pandas
    df = pd.read_csv("./FIFA18_players_database/CompleteDataset.csv", low_memory=False)  # Open file to clean
    df = df.reset_index()  # Make sure indexes pair with number of rows

    cols = list(range(14, 48))
    cols.append(64)
    df = df.iloc[:, cols]  # Extract only columns of interest
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the rows of the
    df = df[~df["Preferred Positions"].str.contains("GK")]  # Remove all goalkeeper rows
    cols_to_drop = [col for col in df.columns if "GK" in col]  # Find all GK attributes
    df.drop(cols_to_drop, axis=1, inplace=True)  # Remove columns
    df = df.map(clean_plus_minus)  # Remove +, -.

    # Calculate 80-20 split of number of rows for training and testing
    num_rows = len(df.index)
    num_train_rows = int(num_rows * 0.8)
    num_test_rows = num_rows - num_train_rows

    # Create training set and test set
    train_df = df.head(num_train_rows)
    test_df = df.tail(num_test_rows)

    # Split rows with multiple labels into multiple rows
    train_df['Preferred Positions'] = train_df['Preferred Positions'].str.strip()  # Remove whitespaces
    train_df["Preferred Positions"] = train_df["Preferred Positions"].str.split(" ")
    train_df = train_df.explode("Preferred Positions")
    X_train = train_df.iloc[:, :-1].to_numpy()  # Only save attributes to X
    X_test = test_df.iloc[:, :-1].to_numpy()
    y_train = train_df.iloc[:, -1].to_numpy()   # Only save last column in y
    y_test = test_df.iloc[:, -1].to_numpy()

    data = (X_train, X_test, y_train, y_test)

    return data