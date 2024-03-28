import numpy as np
from sklearn.decomposition import PCA


def distance_from_center(points):
    if len(points[0]) == 2:
        return np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    elif len(points[0]) == 3:
        return np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 1] ** 2)
    else:
        raise ValueError("Invalido pa")


def apply_pca(dataset: np.array, n_components: int):

    column_means = np.mean(dataset, axis=0)
    standardized_data = dataset - column_means

    pca = PCA(n_components=n_components)
    pca.fit(standardized_data)

    transformed = pca.transform(standardized_data)

    return transformed
