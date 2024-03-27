import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calc_fire_frequency(firings: [np.ndarray], speed_time: np.ndarray, speed: np.ndarray, bin_size: int) -> np.ndarray:
    dt = np.mean(np.diff(speed_time))

    bin_tiempos_velocidades = speed_time[::int(bin_size / dt)]
    # bin_velocidades = [np.mean(speed[i * int(bin_size / dt):(i + 1) * int(bin_size / dt)]) for i in
    #                    range(len(bin_tiempos_velocidades) - 1)]

    tasa_disparo = np.zeros((len(firings), len(bin_tiempos_velocidades) - 1))
    for n in range(tasa_disparo.shape[0]):
        tasa_disparo[n, :], _ = np.histogram(firings[n], bins=bin_tiempos_velocidades)

    return tasa_disparo


def pca_fire_frequency(fire_frequency: np.ndarray):

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(fire_frequency)

    pca = PCA(n_components=len(standardized_data[0]))
    pca.fit(standardized_data)
    transformada = pca.transform(standardized_data)

    varianza_explicada = pca.explained_variance_ratio_

    return transformada, varianza_explicada

