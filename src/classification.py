import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def readData():
    csv_path = os.path.join(os.getcwd(), "data/input_train.csv")
    data = pd.read_csv(csv_path, header=None)
    print(data.info())
    print(data.describe())

    # print(data.describe())
    # print(data.head())
    # sns.pairplot(data, diag_kind="kde", palette="bright")
    # pd.plotting.scatter_matrix(data, figsize=(8, 8))
    # plt.show()

    return data


def clusterData(data, n_clusters=10):
    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(data) for k in range(1, n_clusters)]
    inertias = [model.inertia_ for model in kmeans_per_k]

    # plt.figure(figsize=(8, 3.5))
    # plt.plot(range(1, n_clusters), inertias, "bo-")
    # plt.xlabel("$k$", fontsize=14)
    # plt.ylabel("Inertia", fontsize=14)
    # plt.show()

    silhouette_scores = [silhouette_score(data, model.labels_) for model in kmeans_per_k[1:]]
    plt.plot(range(2, n_clusters), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.tight_layout()
    plt.savefig("images/silhouette.png", format="png", dpi=300)    
    plt.show()


def plot3D(X, y, n_clusters=7):
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    for k in range(n_clusters):
        xs = X[y == k]
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], marker="o", s=2)
    plt.tight_layout()
    plt.savefig("images/scatter3d.png", format="png", dpi=300)    
    plt.show()


if __name__ == "__main__":
    X = readData()

    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X))

    pca = PCA(n_components=0.7)
    X_reduced = pca.fit_transform(X_norm)
    X_recoverd = pca.inverse_transform(X_reduced)
    print(np.cumsum(pca.explained_variance_ratio_))

    # clusterData(X_reduced, n_clusters=12)
    
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    y_pred = kmeans.predict(X_reduced)
    print(kmeans.cluster_centers_)

    plot3D(X_reduced, y_pred, n_clusters=k)
