import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca(dataset: np.array):

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(dataset)

    pca = PCA(n_components=21)
    pca.fit(standardized_data)

    transformada = pca.transform(standardized_data)

    return transformada