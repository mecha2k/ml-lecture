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
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import cm


def readData():
    csv_path = os.path.join(os.getcwd(), "data/input_train.csv")
    train = pd.read_csv(csv_path, header=None)
    print(train.info())
    print(train.describe())

    csv_path = os.path.join(os.getcwd(), "data/input_test.csv")
    test = pd.read_csv(csv_path, header=None)
    print(test.info())

    sns.pairplot(train, diag_kind="kde", palette="bright")
    plt.savefig("deeplearning/images/train_data.png", format="png", dpi=300)
    plt.show()

    return train, test


def saveData(data):
    csv_path = os.path.join(os.getcwd(), "data/output_test.csv")
    data = pd.Series(data)
    data.to_csv(csv_path, index=False)
    print("----------------output value_counts")
    print(data.value_counts())


def runKMeans(data, n_clusters=10):
    kmeans_per_k = [KMeans(n_clusters=_k, random_state=42).fit(data) for _k in range(1, n_clusters)]

    silhouette_scores = [silhouette_score(data, model.labels_) for model in kmeans_per_k[1:]]
    print(np.around(silhouette_scores, 2))

    plt.plot(range(2, n_clusters), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.savefig("deeplearning/images/silhouette_score.png", format="png", dpi=300)
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

    plt.savefig("deeplearning/images/silhouette1.png", format="png", dpi=300)
    plt.show()


def plot3D(_x, y, center, n_clusters, name="scatter3d-1"):
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection="3d")
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))
    for _k in range(n_clusters):
        xs = _x[y == _k]
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], marker="o", s=2, color=colors[_k])
        ax.scatter(center[_k, 0], center[_k, 1], center[_k, 2], marker="*", s=500, color=colors[_k])
    plt.tight_layout()
    plt.savefig(f"deeplearning/images/{name}.png", format="png", dpi=300)
    plt.show()


def runGaussianMixture(X_):
    gms_per_k = [
        GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X_) for k in range(1, 11)
    ]

    bics = [model.bic(X_) for model in gms_per_k]
    aics = [model.aic(X_) for model in gms_per_k]

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
    plt.savefig("deeplearning/images/aic_bic_vs_k_plot.png", format="png", dpi=300)
    plt.show()


def runVariationalBayesianGaussian(X_):
    bgm = BayesianGaussianMixture(
        n_components=10,
        n_init=2,
        max_iter=10000,
        init_params="kmeans",
        weight_concentration_prior=0.01,
        verbose=False,
        random_state=42,
    )
    bgm.fit(X_)
    print(np.round(bgm.weights_, 2))
    print(np.round(bgm.means_, 2))
    print(bgm.converged_)
    print(bgm.n_iter_)


if __name__ == "__main__":
    # read input_train, input_test data
    X, X_test = readData()

    # normalization
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    X_norm_test = scaler.transform(X_test)

    # analyze the optimum number of cluster using KMeans, GaussianMixture
    runKMeans(X_norm)
    runGaussianMixture(X_norm)
    runVariationalBayesianGaussian(X_norm)

    # fix the number of cluster to 4
    clusters = 4

    gm = GaussianMixture(n_components=clusters, n_init=30, random_state=42).fit(X_norm)
    y_pred_gm = gm.predict(X_norm_test)
    center_gm = scaler.inverse_transform(gm.means_)
    print(np.around(center_gm, 1))

    # save output_test.csv
    saveData(y_pred_gm)

    # dimension reduction to 3d to plot scatter distribution
    pca = PCA(n_components=0.7)
    X_reduced = pca.fit_transform(X_norm)
    X_recoverd = pca.inverse_transform(X_reduced)
    print(np.cumsum(pca.explained_variance_ratio_))

    gm = GaussianMixture(n_components=clusters, n_init=30, random_state=42).fit(X_reduced)
    y_pred_gm = gm.predict(X_reduced)
    center_gm = gm.means_

    print(np.around(gm.weights_, 2))
    print(np.around(gm.means_, 2))
    print(gm.converged_, gm.n_iter_)

    plot3D(X_reduced, y_pred_gm, center_gm, n_clusters=clusters, name="scatter-gm")
