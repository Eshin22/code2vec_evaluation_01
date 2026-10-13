"""Runner for Code2Vec clustering following Al-Debagy & Martinek (2021).
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering


def run_code2vec_clustering(class_vectors_path, target_k, seed=42):
    """
    Partitions classes into target_k microservices based on their
    Code2Vec distributed class representations.
    """
    class_vectors = np.load(class_vectors_path)
    N = class_vectors.shape[0]
    target_k = max(2, min(target_k, N))

    # KMeans clustering with seed reproducibility
    kmeans = KMeans(n_clusters=target_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(class_vectors)

    return labels
