import pandas as pd
import re
# TODO: Make a library of each function.
def clean_plus_minus(value):
    if isinstance(value, str):
        return re.sub(r'[+-].*', '',value)
    return value

def engineer_features(df):
    cols = list(range(13, 46))
    cols.append(63)
    df = df.iloc[:, cols]  # Extract only columns of interest
    df = df[~df["Preferred Positions"].str.contains("GK")]  # Remove all goalkeeper rows
    cols_to_drop = [col for col in df.columns if "GK" in col]  # Find all GK attributes
    df.drop(cols_to_drop, axis=1, inplace=True)  # Remove columns
    df = df.map(clean_plus_minus)  # Remove +, -.

    return df

def split_data(df, split):
    pd.options.mode.copy_on_write = True  # Enable copy on write in pandas

    # Calculate 80-20 split of number of rows for training and testing
    num_rows = len(df.index)
    num_train_rows = int(num_rows * 0.8)
    num_test_rows = num_rows - num_train_rows

    # Create training set and test set split
    if split == 0:
        test_df = df[:num_test_rows]
        train_df = df[num_test_rows:]
    elif split == 1:
        test_df = df[num_test_rows:num_test_rows * 2]
        train_df = pd.concat([df[:num_test_rows], df[num_test_rows * 2:]])
    elif split == 2:
        test_df = df[num_test_rows * 2:num_test_rows * 3]
        train_df = pd.concat([df[:num_test_rows * 2], df[num_test_rows * 3:]])
    elif split == 3:
        test_df = df[num_test_rows * 3:num_test_rows * 4]
        train_df = pd.concat([df[:num_test_rows * 3], df[num_test_rows * 4:]])
    else:
        test_df = df[num_test_rows * 4:]
        train_df = df[:num_test_rows * 4]

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