import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('./FIFA18_players_database/clean_data.csv')

# Remove outliers with IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # First quantile
        Q3 = df[col].quantile(0.75)  # Third quantile
        IQR = Q3 - Q1  # IQR distance

        # Identify limits
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Find all numeric values
numeric_columns = df.select_dtypes(include='number').columns

# Remove all outliers
df_cleaned = remove_outliers_iqr(df, numeric_columns)

df_cleaned.to_csv('./FIFA18_players_database/clean_data_W_O_outliers.csv', index=False)

# Normalize with MinMaxScaler
scaler = MinMaxScaler()
df_cleaned[numeric_columns] = scaler.fit_transform(df_cleaned[numeric_columns])

df_cleaned.to_csv('./FIFA18_players_database/clean_data_normalized_W_O_outliers.csv', index=False)

