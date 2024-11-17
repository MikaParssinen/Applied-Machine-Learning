import pandas as pd


def split_data(use_normalized=False):
    # Choose right file depending on argument
    file_path = (
        "./FIFA18_players_database/clean_data_normalized.csv"
        if use_normalized
        else "./FIFA18_players_database/clean_data.csv"
    )
    df = pd.read_csv(file_path, low_memory=False)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the rows of the dataframe
    X = df.iloc[:, :-1]  # Only save attributes to X
    y = df.iloc[:, -1]  # Only save last column in y

    num_rows = len(df.index)
    num_train_rows = int(num_rows * 0.01)
    num_test_rows = num_rows - num_train_rows

    X_train = X.head(num_train_rows)
    X_test = X.tail(num_test_rows)
    y_train = y.head(num_train_rows)
    y_test = y.tail(num_test_rows)

    data = (X_train, X_test, y_train, y_test)

    return data