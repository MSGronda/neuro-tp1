from utils import *
from processing import *
from graphs import *


def ej_1a(data):
    # PCA
    # pca_transformed = apply_pca(data, 3)
    # pca_distances = distance_from_center(pca_transformed)
    # graph_euclidian(pca_transformed, pca_distances)

    # t-SNE
    # tsne_transformed = apply_tsne(data, 3, 30, 200, 1000)
    # tsne_distances = distance_from_center(tsne_transformed)
    # graph_euclidian(tsne_transformed, tsne_distances)

    # ISOMAP
    isomap_transformed = apply_isomap(data, 3, 15)
    isomap_distances = distance_from_center(isomap_transformed)
    graph_euclidian(isomap_transformed, isomap_distances)


if __name__ == "__main__":
    dataset = get_signal_data_by_electrode('./dataset', SOLVING_ARITHMATIC)

    ej_1a(dataset[0].T)




