import os
import numpy as np
import pyedflib


RELAXED = '_1.edf'
SOLVING_ARITHMATIC = '_2.edf'


def get_signal_data_by_electrode(path: str, ends_with: str):
    data = []
    for filename in os.listdir(path):
        if filename.endswith(ends_with):
            d = pyedflib.EdfReader(os.path.join(path, filename))

            patient_data = []
            for i in range(d.signals_in_file):
                patient_data.append(d.readSignal(i))

            d.close()

            data.append(np.array(patient_data))

    return data

