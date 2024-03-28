from utils import *
from processing import *
from graphs import *


def ej_1a(data):
    transformed = apply_pca(data, 3)

    distances = distance_from_center(transformed)

    graph_pca_transformed(transformed, distances)


if __name__ == "__main__":
    dataset = get_signal_data_by_electrode('./dataset', SOLVING_ARITHMATIC)

    ej_1a(dataset[0].T)




