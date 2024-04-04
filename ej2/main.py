from utils import *
from processing import *
from graphs import *


def ej_1a(data):
    # PCA
    pca_transformed = apply_pca(data, 3)
    graph_euclidian(pca_transformed, generate_timeframe(pca_transformed))

    # t-SNE
    tsne_transformed = apply_tsne(data, 3, 30, 200, 1000)
    graph_euclidian(tsne_transformed, generate_timeframe(tsne_transformed))

    # ISOMAP
    isomap_transformed = apply_isomap(data, 3, 15)
    graph_euclidian(isomap_transformed, generate_timeframe(isomap_transformed))


if __name__ == "__main__":
    dataset = get_signal_data_by_electrode('./dataset', SOLVING_ARITHMATIC)

    ej_1a(dataset[0].T)




