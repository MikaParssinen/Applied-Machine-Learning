import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


# Standardize function
def standardize(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df


# Calculate the dunn index of clusters
def dunn_index(data, labels):
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    # If we try to calculate 1 cluster this function is pointless
    if n_clusters < 2:
        raise ValueError("Dunn Index is not defined for less than two clusters.")

    intra_cluster_dist = []
    inter_cluster_dist = []

    # Intra-cluster distance
    for i in unique_clusters:
        cluster_points = data[labels == i]
        if len(cluster_points) > 1:
            intra_cluster_dist.append(np.max(cdist(cluster_points, cluster_points)))
        else:
            intra_cluster_dist.append(0)  # If there is only 1 data point in the cluster

    # Inter-cluster distance
    for i, cluster_i in enumerate(unique_clusters):
        for j, cluster_j in enumerate(unique_clusters):
            if i < j:  # Avoid duplicate computations
                points_i = data[labels == cluster_i]
                points_j = data[labels == cluster_j]
                inter_cluster_dist.append(np.min(cdist(points_i, points_j)))

    return np.min(inter_cluster_dist) / np.max(intra_cluster_dist)

