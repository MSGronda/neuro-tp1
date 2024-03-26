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


def graph_velocity_time(velocity_time: np.ndarray):
    plt.figure(figsize=(10, 6))

    plt.scatter(range(len(velocity_time)), velocity_time, color='blue', label='Velocidad vs Tiempo')

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocity (???)')
    plt.title('Velocidad vs Tiempo')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graph_firing_frequency(firing_frequency, neuron_indices):
    plt.figure(figsize=(10, 6))

    for f, i in zip(firing_frequency, neuron_indices):
        plt.scatter(range(len(f)), f, label=f'Neuron {i}')

    plt.xlabel('Time bin')
    plt.ylabel('Number of firings')
    plt.title('Number of firings vs Time bin')
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

