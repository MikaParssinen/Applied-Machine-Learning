import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# Standardize function
def standardize(df, retscaler=None):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    if retscaler:
        return scaled_df, scaler
    else:
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


def remove_outliers_iqr(df, bound):
    # Find numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    new_df = df.copy()

    for col in numeric_cols:
        Q1 = new_df[col].quantile(bound)
        Q3 = new_df[col].quantile(1.00 - bound)
        IQR = Q3 - Q1

        # Define the bounds for detecting outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outliers
        new_df = new_df[(new_df[col] >= lower_bound) & (new_df[col] <= upper_bound)]

    return new_df


def pca(df, dim):
    if dim == 2:
        n_component = 2
        columns = ['PC1', 'PC2']
    elif dim == 3:
        n_component = 3
        columns = ['PC1', 'PC2', 'PC3']
    else:
        raise Exception('Choose a valid dimension')

    pca = PCA(n_component)
    pca_components = pca.fit_transform(df.drop(columns=['Cluster']))
    pca_df = pd.DataFrame(pca_components, columns=columns)
    pca_df['Cluster'] = df['Cluster']  # Add the cluster labels

    return pca_df


def tsne(df, dim):
    if dim == 2:
        n_component = 2
        columns = ['TSNE1', 'TSNE2']
    elif dim == 3:
        n_component = 3
        columns = ['TSNE1', 'TSNE2', 'TSNE3']
    else:
        raise Exception('Choose a valid dimension')

    tsne = TSNE(n_components=n_component, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(df.drop(columns=['Cluster']))

    # Convert results into a DataFrame
    tsne_df = pd.DataFrame(tsne_results, columns=columns)
    tsne_df['Cluster'] = df['Cluster']

    return tsne_df

def generate_silhouette_scores(n_list: list, df: pd.DataFrame, model: str) -> list[float]:
    silhouette_scores = []
    for n in n_list:
        if model == "agglomerative":
            silhouette_scores.append(
                silhouette_score(df, AgglomerativeClustering(n_clusters=n).fit_predict(df)))
        else:
            db = DBSCAN(eps=n, min_samples=17).fit(df)
            silhouette_scores.append(silhouette_score(df, db.labels_))
    return silhouette_scores

def bar_plot(x_values, y_values, x_label=None, y_label=None, title=None, size=None, rotation=None, ylim=None, width=None):
    if size:
        plt.figure(figsize=size)
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label, fontsize=10)
    if y_label:
        plt.ylabel(y_label, fontsize=10)
    if rotation:
        plt.xticks(rotation=rotation)
    if ylim:
        plt.ylim(ylim)
    if width:
        plt.bar(x_values, y_values, width=width)
    else:
        plt.bar(x_values, y_values)
    plt.show()



def unstandardize(df, scaler):
    df_unstandardized = scaler.inverse_transform(df.drop(columns=['Cluster']))

    df_unstandardized = pd.DataFrame(df_unstandardized, columns=df.drop(columns=['Cluster']).columns)

    df_unstandardized['Cluster'] = df['Cluster']

    return df_unstandardized


def calculate_kn_distance(X,k):

    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(
                math.sqrt(
                    ((X[i,0] - X[j,0]) ** 2) +
                    ((X[i,1] - X[j,1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    return kn_distance

