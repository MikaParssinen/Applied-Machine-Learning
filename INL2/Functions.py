import pandas as pd
from sklearn.preprocessing import StandardScaler


# Standardize function
def standardize(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df

