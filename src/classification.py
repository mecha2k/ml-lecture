import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import cm


def readData():
    csv_path = os.path.join(os.getcwd(), "data/input_train.csv")
    data = pd.read_csv(csv_path, header=None)
    data = data[:1000]
    print(data.info())
    print(data.describe())

    # sns.pairplot(data, diag_kind="kde", palette="bright")
    # pd.plotting.scatter_matrix(data, figsize=(8, 8))
    # plt.show()

    return data


def analyzeCluster(data, n_clusters):
    kmeans_per_k = [KMeans(n_clusters=_k, random_state=42).fit(data) for _k in range(1, n_clusters)]

    # inertias = [model.inertia_ for model in kmeans_per_k]
    # plt.figure(figsize=(8, 3.5))
    # plt.plot(range(1, n_clusters), inertias, "bo-")
    # plt.xlabel("$k$", fontsize=14)
    # plt.ylabel("Inertia", fontsize=14)
    # plt.show()

    silhouette_scores = [silhouette_score(data, model.labels_) for model in kmeans_per_k[1:]]
    print(silhouette_scores)

    plt.plot(range(2, n_clusters), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.tight_layout()
    plt.savefig("images/silhouette.png", format="png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 8))

    for k in range(2, n_clusters):
        ax = plt.subplot(3, 3, k - 1)
        ax.set_xlim(0, 1)

        y_pred = kmeans_per_k[k - 1].labels_
        silhouette_coefficients = silhouette_samples(data, y_pred)

        padding = len(data) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()
            color = cm.Spectral(i / k)
            plt.fill_betweenx(
                np.arange(pos, pos + len(coeffs)),
                0,
                coeffs,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (3, 6, 9):
            plt.ylabel("Cluster")
        if k in (9, 10, 11):
            plt.xlabel("Silhouette Coefficient")
        # plt.gca().set_xticks([-0.2, 0, 0.4, 0.8, 1.2])
        # plt.tick_params(labelbottom=False)
        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=14)
        plt.tight_layout()

    plt.savefig("images/silhouette1.png", format="png", dpi=300)
    plt.show()


def plot3D(_x, y, center, n_clusters=7):
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection="3d")
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    for _k in range(n_clusters):
        xs = _x[y == _k]
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], marker="o", s=2, color=colors[_k])
        ax.scatter(center[_k, 0], center[_k, 1], center[_k, 2], marker="*", s=500, color=colors[_k])
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

    analyzeCluster(X_reduced, n_clusters=10)

    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    y_pred = kmeans.predict(X_reduced)
    print(kmeans.cluster_centers_)

    plot3D(X_reduced, y_pred, kmeans.cluster_centers_, n_clusters=k)
