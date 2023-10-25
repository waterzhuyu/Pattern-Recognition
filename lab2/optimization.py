"""Optimization for k-neighbor method. """
import numpy as np


def knn_density_estimation(data, x_values, k_neighbor):
    t = k_neighbor / len(data)
    prob = []
    for x in x_values:
        distances = [abs(x-s) for s in data]
        distances.sort()
        v = distances[k_neighbor] * 2
        prob.append(t / v)
    return np.array(prob)
