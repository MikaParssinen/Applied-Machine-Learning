# Library for all the functions.
import pandas as pd
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Normalize function
def normalize(df, method, complete_data):
    if method == 'minmax':
        scalar = MinMaxScaler()
    else:
        scalar = StandardScaler()

    if complete_data:
        df.iloc[:, :-1] = scalar.fit_transform(df.iloc[:, :-1])
        return df
    else:
        new_df = scalar.fit_transform(df)
        return new_df

# Outlier removal function
def remove_outliers(df, percentile):
    # Remove outliers
    new_df = df.copy()
    cols = new_df.columns.tolist()
    cols.remove("Preferred Positions")
    for col in cols:  # For each feature in the set, remove each row that has at least one feature that is
                             # below the 1-percentile or it is above the 99th percentile
        new_df[col] = new_df[col].astype(int)
        q_u = new_df[col].quantile(1 - (percentile/100))
        q_l = new_df[col].quantile(percentile / 100)
        new_df = new_df[new_df[col] <= q_u]
        new_df = new_df[new_df[col] >= q_l]
    return new_df


def remove_outliers_iqr(df):
    # Find numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    new_df = df.copy()

    for col in numeric_cols:
        Q1 = new_df[col].quantile(0.25)
        Q3 = new_df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the bounds for detecting outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outliers
        new_df = new_df[(new_df[col] >= lower_bound) & (new_df[col] <= upper_bound)]

    return new_df

# Function to split data into train and test sets
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
    X_train = train_df.iloc[:, :-1].reset_index(drop=True)  # Only save attributes to X
    X_test = test_df.iloc[:, :-1].reset_index(drop=True)
    y_train = train_df.iloc[:, -1].reset_index(drop=True)   # Only save last column in y
    y_test = test_df.iloc[:, -1].reset_index(drop=True)

    data = (X_train, X_test, y_train, y_test)

    return data

# Accuracy score function
def accuracy_scorer(y_test, y_pred):
    length = len(y_test)
    count = 0
    for i in range(length):
        if y_pred[i] in y_test[i]:
            count += 1
    return count/length

# Remove unwanted + and - in features
def clean_plus_minus(value):
    if isinstance(value, str):
        return re.sub(r'[+-].*', '',value)
    return value

# Remove unwanted columns and rows
def engineer_features(df):
    cols = list(range(13, 46))
    cols.append(63)
    df = df.iloc[:, cols]  # Extract only columns of interest
    df = df[~df["Preferred Positions"].str.contains("GK")]  # Remove all goalkeeper rows
    cols_to_drop = [col for col in df.columns if "GK" in col]  # Find all GK attributes
    df.drop(cols_to_drop, axis=1, inplace=True)  # Remove columns
    df = df.map(clean_plus_minus)  # Remove +, -.

    df.reset_index(drop=True, inplace=True)

    return df

# Function to run gaussian naive bayes
def run_naive_bayes(X_train, X_test, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    return y_test_pred, y_train_pred

# Function to run SVM
def run_SVM_classifier(X_train, X_test, y_train, kernel, c, retClass):
    svm_classifier = SVC(kernel=kernel, C=c, gamma='auto')
    svm_classifier.fit(X_train, y_train)
    y_test_pred = svm_classifier.predict(X_test)
    y_train_pred = svm_classifier.predict(X_train)

    if retClass:
        return y_test_pred, y_train_pred, svm_classifier
    else:
        return y_test_pred, y_train_pred

def run_RF(n_est, X_train, X_test, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=n_est, min_samples_split=10, min_samples_leaf=4)
    rf_classifier.fit(X_train, y_train)

    y_test_pred = rf_classifier.predict(X_test)
    y_train_pred = rf_classifier.predict(X_train)




    return y_test_pred, y_train_pred
