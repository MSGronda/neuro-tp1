import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def bin_firings(firings: [np.ndarray], speed_time: np.ndarray, bin_size: int):
    dt = np.mean(np.diff(speed_time))

    speed_time_bins = speed_time[::int(bin_size / dt)]

    binned_firings = np.zeros((len(firings), len(speed_time_bins) - 1))
    for n in range(binned_firings.shape[0]):
        binned_firings[n, :], _ = np.histogram(firings[n], bins=speed_time_bins)

    return binned_firings


def bin_speed(speed_time: np.ndarray, speed: np.ndarray, bin_size: int):

    dt = np.mean(np.diff(speed_time))
    bin_tiempos_velocidades = speed_time[::int(bin_size / dt)]

    return np.array([np.mean(speed[i * int(bin_size / dt):(i + 1) * int(bin_size / dt)]) for i in
                        range(len(bin_tiempos_velocidades) - 1)])


def pca_fire_frequency(fire_frequency: np.ndarray):

    column_means = np.mean(fire_frequency, axis=0)
    standardized_data = fire_frequency - column_means

    pca = PCA(n_components=len(standardized_data[0]))
    pca.fit(standardized_data)
    transformed = pca.transform(standardized_data)

    return transformed, pca.explained_variance_ratio_, pca.components_

