import matplotlib.pyplot as plt


def graph_euclidian(transformed, distances):

    if len(transformed[0]) == 2:
        fig, ax = plt.subplots()
        ax.scatter(transformed[:, 0], transformed[:, 1], c=distances)
    elif len(transformed[0]) == 3:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=distances)
    else:
        raise ValueError("Invalido pa")

    plt.tight_layout()
    plt.show()


