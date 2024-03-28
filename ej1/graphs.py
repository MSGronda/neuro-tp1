import matplotlib.pyplot as plt
import numpy as np


def graph_firings(firings: [np.ndarray]):
    plt.figure(figsize=(10, 6))

    for i, timestamps in enumerate(firings):
        plt.eventplot(timestamps, lineoffsets=i + 1, linewidths=1)

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Disparos')
    plt.title('Trenes de Disparo vs Tiempo')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def graph_speed_vs_time(speed, speed_time):
    plt.figure(figsize=(10, 6))

    plt.scatter(speed_time, speed, color='blue', label='Velocidad vs Tiempo')

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (m/s)')
    plt.title('Velocidad vs Tiempo')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graph_firing_frequency(firing_frequency, binned_speed, neuron_index):
    plt.figure(figsize=(10, 6))

    plt.scatter(binned_speed, firing_frequency, label=f'Neurona {neuron_index}')

    plt.xlabel('Velocidades agrupadas')
    plt.ylabel('Disparos agrupados')
    plt.title(f'Disparos agrupados vs Velocidades agrupadas (para neurona {neuron_index})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graph_cumulative_variance(explained_variance):
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.plot(np.arange(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Número de componentes principales')
    plt.ylabel('Varianza explicada acumulativa')
    plt.title('Varianza explicada acumulativa por número de componentes principales')
    plt.grid(True)
    plt.show()


def graph_pcn_vs_other(pcn, other, pcn_name: str, other_name: str):
    plt.scatter(other, pcn)
    plt.xlabel(other_name)
    plt.ylabel(pcn_name)
    plt.title(f'{pcn_name} vs {other_name}')
    plt.show()

