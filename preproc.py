# preprocessing of raw ECG signals
# each record is transformed into an 8D vector of:
# signal mean, variance
# distribution of quartiles
# gradient magnitude (roughness) of the signal
# energy
# spectral entropy

import wfdb
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
import networkx as nx


#############################
#    FEATURE EXTRACTION     #
#############################

def extract_features(signal):
    import numpy as np

    x = signal[:, 0].astype(float)

    # remove DC drift by subtracting signal mean from signal
    x = x - np.mean(x)

    # basic stat declaration
    mean = np.mean(x)
    std = np.std(x)

    # quartile distribution (used in lieu of minmax)
    q25 = np.percentile(x, 25)
    q50 = np.median(x)
    q75 = np.percentile(x, 75)

    # waveform roughness is calculated using the mean of the
    # absolute value of the difference of consecutive values
    roughness = np.mean(np.abs(np.diff(x)))

    # energy (signal power)
    energy = np.sum(x ** 2) / len(x)

    # convert signal into freq. domain & ignore phase info
    # to capture freq-domain independent of amplitude
    # uses shannon entropy to tell apart
    # 1. low entropy i.e structured signal
    # 2. high entropy i.e noise
    fft = np.abs(np.fft.rfft(x))
    fft = fft / (np.sum(fft) + 1e-12)
    spectral_entropy = -np.sum(fft * np.log(fft + 1e-12))

    return np.array([
        mean,
        std,
        q25,
        q50,
        q75,
        roughness,
        energy,
        spectral_entropy
    ])

#############################
#    DATASET LOADING        #
#############################

def load_ecg_matrix(path):
    records = sorted([                              # get all .dat
        f.split('.')[0]
        for f in os.listdir(path)
        if f.endswith(".dat")
    ])

    X = []
    names = []

    for r in records:
        full_path = os.path.join(path, r)
        rec = wfdb.rdrecord(full_path)              # load waveform thru wfdb

        X.append(extract_features(rec.p_signal))    # convert signal to feature vect
        names.append(r)

    X = np.vstack(X)                                # convert list to matrix
    return X, names                                 # sample # x features


#############################
#    COMSTRUCT GRAPH        #
#############################

def build_graph(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X) # find kNNs in feature space
    _, idx = nbrs.kneighbors(X)

    G = nx.Graph()

    for i in range(len(X)):
        for j in idx[i]:
            if i != j:
                if i in idx[j]:         # consider only mutual kNN
                    G.add_edge(i, j)

    return G