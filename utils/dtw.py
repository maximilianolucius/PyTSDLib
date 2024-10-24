__author__ = 'Brian Iwana'

import numpy as np
import sys

RETURN_VALUE = 0
RETURN_PATH = 1
RETURN_ALL = -1

# Core DTW function for traceback
def _traceback(DTW, slope_constraint):
    """
    Trace back the optimal path for DTW based on the slope constraint.
    :param DTW: The accumulated cost matrix.
    :param slope_constraint: The slope constraint, either 'asymmetric' or 'symmetric'.
    :return: Two arrays representing the traced path.
    """
    i, j = np.array(DTW.shape) - 1
    p, q = [i-1], [j-1]

    if slope_constraint == "asymmetric":
        while i > 1:
            tb = np.argmin((DTW[i-1, j], DTW[i-1, j-1], DTW[i-1, j-2]))
            if tb == 0:
                i -= 1
            elif tb == 1:
                i -= 1
                j -= 1
            elif tb == 2:
                i -= 1
                j -= 2
            p.insert(0, i-1)
            q.insert(0, j-1)

    elif slope_constraint == "symmetric":
        while i > 1 or j > 1:
            tb = np.argmin((DTW[i-1, j-1], DTW[i-1, j], DTW[i, j-1]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            elif tb == 2:
                j -= 1
            p.insert(0, i-1)
            q.insert(0, j-1)
    else:
        sys.exit(f"Unknown slope constraint {slope_constraint}")

    return np.array(p), np.array(q)

def dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None):
    """
    Computes the Dynamic Time Warping (DTW) distance between two sequences.
    :param prototype: Prototype sequence.
    :param sample: Sample sequence.
    :param return_flag: Flag to control the return value (value/path/all).
    :param slope_constraint: The slope constraint, either 'asymmetric' or 'symmetric'.
    :param window: Sakoe-Chiba window size.
    :return: DTW value, path, or all based on return_flag.
    """
    p, s = prototype.shape[0], sample.shape[0]
    assert p != 0 and s != 0, "Prototype or Sample is empty!"

    if window is None:
        window = s

    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window) + 1
        cost[i, start:end] = np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    DTW = _cumulative_matrix(cost, slope_constraint, window)

    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]

def _cumulative_matrix(cost, slope_constraint, window):
    """
    Computes the cumulative cost matrix for DTW.
    :param cost: The cost matrix.
    :param slope_constraint: The slope constraint, either 'asymmetric' or 'symmetric'.
    :param window: Sakoe-Chiba window size.
    :return: The cumulative cost matrix.
    """
    p, s = cost.shape
    DTW = np.full((p+1, s+1), np.inf)
    DTW[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        for i in range(1, p + 1):
            if i <= window + 1:
                DTW[i, 1] = cost[i-1, 0] + min(DTW[i-1, 0], DTW[i-1, 1])
            for j in range(max(2, i - window), min(s, i + window) + 1):
                DTW[i, j] = cost[i-1, j-1] + min(DTW[i-1, j-2], DTW[i-1, j-1], DTW[i-1, j])
    elif slope_constraint == "symmetric":
        for i in range(1, p + 1):
            for j in range(max(1, i - window), min(s, i + window) + 1):
                DTW[i, j] = cost[i-1, j-1] + min(DTW[i-1, j-1], DTW[i, j-1], DTW[i-1, j])
    else:
        sys.exit(f"Unknown slope constraint {slope_constraint}")

    return DTW

def shape_dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None, descr_ratio=0.05):
    """
    Computes ShapeDTW distance between two sequences.
    :param prototype: Prototype sequence.
    :param sample: Sample sequence.
    :param return_flag: Flag to control the return value (value/path/all).
    :param slope_constraint: The slope constraint, either 'asymmetric' or 'symmetric'.
    :param window: Sakoe-Chiba window size.
    :param descr_ratio: Descriptor ratio for feature length.
    :return: ShapeDTW value, path, or all based on return_flag.
    """
    p, s = prototype.shape[0], sample.shape[0]
    assert p != 0 and s != 0, "Prototype or Sample is empty!"

    if window is None:
        window = s

    p_feature_len = np.clip(np.round(p * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s * descr_ratio), 5, 100).astype(int)

    # Padding
    prototype_pad = np.pad(prototype, ((p_feature_len // 2, p_feature_len // 2), (0, 0)), mode="edge")
    sample_pad = np.pad(sample, ((s_feature_len // 2, s_feature_len // 2), (0, 0)), mode="edge")

    cost = np.full((p, s), np.inf)
    for i in range(p):
        for j in range(max(0, i - window), min(s, i + window)):
            cost[i, j] = np.linalg.norm(sample_pad[j:j + s_feature_len] - prototype_pad[i:i + p_feature_len])

    DTW = _cumulative_matrix(cost, slope_constraint=slope_constraint, window=window)

    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]

# Plotting helpers
def draw_graph2d(cost, DTW, path, prototype, sample):
    """
    Draws a 2D representation of the DTW cost, cumulative matrix, and path.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))

    # Cost matrix
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim(-0.5, cost.shape[0] - 0.5)
    plt.ylim(-0.5, cost.shape[1] - 0.5)

    # Cumulative matrix
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0] + 1, path[1] + 1, 'y')
    plt.xlim(-0.5, DTW.shape[0] - 0.5)
    plt.ylim(-0.5, DTW.shape[1] - 0.5)

    # Prototype
    plt.subplot(2, 3, 4)
    plt.plot(prototype[:, 0], prototype[:, 1], 'b-o')

    # Connection between points
    plt.subplot(2, 3, 5)
    for i in range(path[0].shape[0]):
        plt.plot([prototype[path[0][i], 0], sample[path[1][i], 0]], [prototype[path[0][i], 1], sample[path[1][i], 1]], 'y-')
    plt.plot(sample[:, 0], sample[:, 1], 'g-o')
    plt.plot(prototype[:, 0], prototype[:, 1], 'b-o')

    # Sample
    plt.subplot(2, 3, 6)
