import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


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

def generate_silhouette_scores(n_list: list[int], df: pd.DataFrame, model: str) -> list[float]:
    silhouette_scores = []
    for n in n_list:
        if model == "agglomerative":
            silhouette_scores.append(
                silhouette_score(df, AgglomerativeClustering(n_clusters=n).fit_predict(df)))
        else:
            pass
    return silhouette_scores

def bar_plot(x_values, y_values, x_label=None, y_label=None, title=None, size=None, rotation=None):
    if size:
        plt.figure(figsize=size)
    if title:
        plt.title(title)
    plt.bar(x_values, y_values)
    if x_label:
        plt.xlabel(x_label, fontsize=10)
    if y_label:
        plt.ylabel(y_label, fontsize=10)
    if rotation:
        plt.xticks(rotation=rotation)
    plt.show()