▚▚▘ Modularity calculation exercise using ECG data ▞▞▖

A small Python project I made to learn what modularity is. I used it to explore graph construction and community detection on ECG waveform data.

Each ECG recording is transformed into a fixed-length feature vector. These vectors form a mutual k-nearest-neighbour graph (where edges=similarity). Thhen each graph is partitioned using the Louvain algorithm and evaluated using a 1:1 implementation of the modularity equation. This is done to the detriment of efficiency because my priority was to learn the equation itself :)

▚▚▘ STRUCTURE ▞▞▖

preproc.py: Preprocessing. ECG recordings are loaded and the waveform features are extracted. Then the kNN graph is constructed.

modularity.py: Louvain algo is used to detect communities and then modularity is computed directly from its mathematical definition

experiment.py: computes and visualises modularity (Q), community count (C) and density (D) change as k (how large each node's local neighbourhood is) varies

▚▚▘ EXTRACTED FEATURES ▞▞▖

* mean
* Standard deviation
* 25th percentile
* median
* 75th percentile
* waveform roughness (avg height b/w adjacent points)
* signal energy (avg strength of the signal)
* spectral entropy (computed using fast fourier -> normalised -> measured using shannon entropy)

▚▚▘ DEPENDENCIES ▞▞▖

numpy matplotlib networkx scikit-learn wfdb python-louvain

▚▚▘ HOW TO USE ▞▞▖

Run experiment.py by providing a WFDB dataset:

python experiment.py <dataset_path>

▚▚▘ OUTPUT ▞▞▖

For each k∈{1,...,n} where k controls the kNN graph construction (by deciding its neighbourhood size), the following are computed:

* modularity (Q) of Louvain communities
* number of detected communities
* graph density

Quantities are minmax normalised and plotted together to show how graph structure changes as connectivity increases.
