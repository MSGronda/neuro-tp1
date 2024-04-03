import pickle
from ej1.graphs import *
from ej1.processing import *


def ej_a(firings, speed, speed_time):
    graph_firings(firings)
    graph_speed_vs_time(speed, speed_time)


def ej_b_c(firings, speed_time, speed):
    firing_frequency = bin_firings(firings, speed_time, 1)
    binned_speed = bin_speed(speed_time, speed, 1)

    indices = [44, 207, 331, 643, 656, 660, 699, 779]

    for i in indices:
        graph_firing_frequency(firing_frequency[i], binned_speed, i)


def ej_d(firings, speed_time):
    firing_frequency = bin_firings(firings, speed_time, 1)

    _, explained_variance, _ = pca_fire_frequency(firing_frequency)

    graph_cumulative_variance(explained_variance)


def ej_e(firings, speed_time, speed):
    firing_frequency = bin_firings(firings, speed_time, 1)
    binned_speeds = bin_speed(speed_time, speed, 1)

    _, _, components = pca_fire_frequency(firing_frequency)

    pc1 = components[0]
    pc2 = components[1]

    graph_pcn_vs_other(pc1, [i for i in range(0, len(pc1))], "PC1", "Tiempo")
    graph_pcn_vs_other(pc1, pc2, "PC1", "PC2")
    graph_pcn_vs_other(pc1, binned_speeds, "PC1", "Velocidad")


if __name__ == '__main__':
    with open('./DataTP1.pkl', 'rb') as fp:
        data = pickle.load(fp)

    firings = data['tiempos_disparos']
    speed_time = data['tiempos_velocidades']
    speed = data['velocidades']

    ej_a(firings, speed, speed_time)
    ej_b_c(firings, speed_time, speed)
    ej_d(firings, speed_time)
    ej_e(firings, speed_time, speed)



