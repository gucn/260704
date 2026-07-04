import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from preproc import load_ecg_matrix, build_graph
from modularity import modularity, louvain_partition


def run_experiment(data_path, max_k=None):

    X, names = load_ecg_matrix(data_path)	# load ecg data to feature matrix

    n = len(X)
    if max_k is None:					 	# graph connectivity defined by k range
        max_k = n

    k_vals = range(1, max_k + 1)

    Q_vals = []								# modularity per k
    C_vals = []								# communities per k
    D_vals = []								# density per k

    #############################
	#      PHASE SPACE SWEEP    #
	#############################

    for k in k_vals:
        G = build_graph(X, k=k)				# rebuild graph for each k val

        communities = louvain_partition(G)	# find communities thru louvain
        Q = modularity(G, communities)		# compute modularity

        Q_vals.append(Q)
        C_vals.append(len(communities))
        D_vals.append(nx.density(G))

        print(f"k={k}, Q={Q}, communities={len(communities)}")

    #############################
	#      NORMALISATION        #
	#############################

    Q_vals = np.array(Q_vals)
    C_vals = np.array(C_vals)
    D_vals = np.array(D_vals)

    Q_norm = (Q_vals - Q_vals.min()) / (Q_vals.max() - Q_vals.min() + 1e-12)
    C_norm = (C_vals - C_vals.min()) / (C_vals.max() - C_vals.min() + 1e-12)
    D_norm = (D_vals - D_vals.min()) / (D_vals.max() - D_vals.min() + 1e-12)

    #############################
	#      VISUALISATION        #
	#############################

    plt.plot(list(k_vals), Q_norm, label="Modularity Q", color="red")
    plt.plot(list(k_vals), C_norm, label="Community count", color="green")
    plt.plot(list(k_vals), D_norm, label="Density", color="blue")

    plt.xlabel("k (kNN parameter)")
    plt.ylabel("Normalised value")
    plt.title("Graph structure vs k (Louvain)")
    plt.legend()
    plt.show()

# warning incase user does not enter the data path
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python experiment.py <data_path>")
        sys.exit(1)

    run_experiment(sys.argv[1])