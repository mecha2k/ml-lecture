import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import cm


def readData():
    csv_path = os.path.join(os.getcwd(), "data/input_train.csv")
    data = pd.read_csv(csv_path, header=None)
    # data = data[:1000]
    # print(data.info())
    # print(data.describe())

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

    for k_ in range(2, n_clusters):
        ax = plt.subplot(3, 3, k_ - 1)
        ax.set_xlim(0, 1)

        y_pred_ = kmeans_per_k[k_ - 1].labels_
        silhouette_coefficients = silhouette_samples(data, y_pred_)

        padding = len(data) // 30
        pos = padding
        ticks = []
        for i in range(k_):
            coeffs = silhouette_coefficients[y_pred_ == i]
            coeffs.sort()
            color = cm.Spectral(i / k_)
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
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k_)))
        if k_ in (3, 6, 9):
            plt.ylabel("Cluster")
        if k_ in (9, 10, 11):
            plt.xlabel("Silhouette Coefficient")
        # plt.gca().set_xticks([-0.2, 0, 0.4, 0.8, 1.2])
        # plt.tick_params(labelbottom=False)
        plt.axvline(x=silhouette_scores[k_ - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k_), fontsize=14)
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
    # plt.show()


def runGaussianMixture(X):
    gms_per_k = [
        GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X) for k in range(1, 11)
    ]

    bics = [model.bic(X) for model in gms_per_k]
    aics = [model.aic(X) for model in gms_per_k]

    plt.figure(figsize=(8, 3))
    plt.plot(range(1, 11), bics, "bo-", label="BIC")
    plt.plot(range(1, 11), aics, "go--", label="AIC")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Information Criterion", fontsize=14)
    plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
    plt.annotate(
        "Minimum",
        xy=(4, bics[3]),
        xytext=(0.35, 0.6),
        textcoords="figure fraction",
        fontsize=14,
        arrowprops=dict(facecolor="black", shrink=0.1),
    )
    plt.legend()
    plt.savefig("./images/aic_bic_vs_k_plot.png", format="png", dpi=300)
    plt.show()


def runVariationalBayesianGaussian(X):
    bgm = BayesianGaussianMixture(
        n_components=12,
        n_init=2,
        max_iter=10000,
        init_params="kmeans",
        weight_concentration_prior=0.01,
        verbose=False,
        random_state=42,
    )
    bgm.fit(X)
    print(bgm.weights_)
    print(np.round(bgm.weights_, 2))
    # print(bgm.means_)
    # print(bgm.covariances_)
    print(bgm.converged_)
    print(bgm.n_iter_)


if __name__ == "__main__":
    X = readData()

    scaler = StandardScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X))

    pca = PCA(n_components=0.7)
    X_reduced = pca.fit_transform(X_norm)
    X_recoverd = pca.inverse_transform(X_reduced)
    # print(np.cumsum(pca.explained_variance_ratio_))

    # analyzeCluster(X_reduced, n_clusters=10)
    runGaussianMixture(X_norm)
    runVariationalBayesianGaussian(X_norm)

    # kmeans = KMeans(n_clusters=3, random_state=42).fit(X_reduced)
    # y_pred = kmeans.predict(X_reduced)
    # print(kmeans.cluster_centers_)

    # gm = GaussianMixture(n_components=3, n_init=30, random_state=42)
    # gm.fit(X)

    # print(gm.means_)
    # print(gm.weights_)
    # print(gm.covariances_)
    # print(gm.converged_)
    # print(gm.n_iter_)
    # print(gm.predict(X))
    # score = gm.score_samples(X)
    # print(score)

    # plot3D(X_reduced, y_pred, kmeans.cluster_centers_, n_clusters=3)
