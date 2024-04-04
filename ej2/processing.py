import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler


def distance_from_center(points):
    if len(points[0]) == 2:
        return np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    elif len(points[0]) == 3:
        return np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    else:
        raise ValueError("Invalido pa")


def generate_timeframe(points):
    return [i for i in range(len(points))]


def apply_pca(dataset: np.array, n_components: int):
    column_means = np.mean(dataset, axis=0)
    standardized_data = dataset - column_means

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(standardized_data)

    pca = PCA(n_components=n_components)

    transformed = pca.fit_transform(standardized_data)

    return transformed


def apply_tsne(dataset: np.array, n_components: int, perplexity: int, learning_rate: int, n_iter: int):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(dataset)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)

    transformed = tsne.fit_transform(standardized_data)

    return transformed


def apply_isomap(dataset: np.array, n_components: int, n_neighbors: int):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(dataset)

    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)

    transformed = isomap.fit_transform(standardized_data)

    return transformed
