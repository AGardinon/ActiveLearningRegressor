#!

import random
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Union, List, Tuple

# --------------------------------------------------------------
# Sampling functions

def sample_landscape(X_landscape: np.ndarray, n_points: int, sampling_mode: str='fps', **kwargs) -> np.ndarray:
    """
    Sample `n_points` from `X_landscape` using the specified `sampling_mode`.

    Args:
        X_landscape: (N, d) array of data points
        n_points: number of points to select
        sampling_mode: 'fps' | 'voronoi' | 'random'
        **kwargs: additional arguments for the sampling method

    Returns:
        indices of selected samples in X_landscape
    """

    sampling_mode_dict = {
        'fps' : fps,
        'voronoi' : voronoi,
        'random' : rnd
    }

    return sampling_mode_dict[sampling_mode](X=X_landscape, n_points=n_points, **kwargs)


def rnd(X: np.ndarray, n_points: int) -> np.ndarray:
    """
    Random sampling to select `n_points` samples from `X`.

    Args:
        X: (N, d) array of data points
        n_points: number of points to select

    Returns:
        indices of selected samples in X
    """

    indices_pool = np.arange(len(X))

    if not isinstance(indices_pool, list):
        indices_pool = list(indices_pool)

    return random.sample(indices_pool, n_points)


def fps(X: np.ndarray, 
        n_points: int, 
        start_idx: int=None, 
        return_distD: bool=False) -> Union[List[int], Tuple[List[int], np.ndarray]]:
    """
    Farthest Point Sampling (FPS) algorithm to select `n_points` samples from `X`.

    Args:
        X: (N, d) array of data points
        n_points: number of points to select
        start_idx: optional starting index for FPS
        return_distD: if True, also return distances of selected points

    Returns:
        indices of selected samples in X
        (optionally) distances of selected points
    """

    if isinstance(X, pd.DataFrame):
        X = np.array(X)

    # init the output quantities
    fps_ndxs = np.zeros(n_points, dtype=int)
    distD = np.zeros(n_points)

    # check for starting index
    if not start_idx:
        # the b limits has to be decreaed because of python indexing
        # start from zero
        start_idx = random.randint(a=0, b=X.shape[0]-1)
    # inset the first idx of the sampling method
    fps_ndxs[0] = start_idx

    # compute the distance from selected point vs all the others
    dist1 = np.linalg.norm(X - X[start_idx], axis=1)

    # loop over the distances from selected starter
    # to find the other n points
    for i in range(1, n_points):
        # get and store the index for the max dist from the point chosen
        fps_ndxs[i] = np.argmax(dist1)
        distD[i - 1] = np.amax(dist1)

        # compute the dists from the newly selected point
        dist2 = np.linalg.norm(X - X[fps_ndxs[i]], axis=1)
        # takes the min from the two arrays dist1 2
        dist1 = np.minimum(dist1, dist2)

        # little stopping condition
        if np.abs(dist1).max() == 0.0:
            print(f"Only {i} iteration possible")
            return fps_ndxs[:i], distD[:i]
        
    if return_distD:
        return list(fps_ndxs), distD
    else:
        return list(fps_ndxs)
    

def voronoi(X: np.ndarray, n_points: int, mode='MiniBatchKMeans') -> np.ndarray:
    """
    Select `n_points` samples from `X` using KMeans clustering and choosing
    the samples closest to the cluster centroids.

    Args:
        X: (N, d) array of data points
        n_points: number of points to select
        mode: 'KMeans' | 'MiniBatchKMeans' for clustering algorithm
    Returns:
        indices of selected samples in X
    """
    
    cluster_mode = {
        'KMeans' : KMeans(init="k-means++", n_clusters=n_points),
        'MiniBatchKMeans' : MiniBatchKMeans(init="k-means++", n_clusters=n_points),
    }

    kmeans = cluster_mode[mode].fit(X)

    # select the k samples closest to their respective cluster centroid
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    return closest


def select_centers_by_distance(
    centers: np.ndarray,
    X_train: np.ndarray,
    threshold: float,
    mode: str = "filter",
    sort_scores: np.ndarray = None,
    tol_dedup: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of centers enforcing a minimum distance to existing training
    points and between selected centers.

    Args:
        centers: (M, d) candidate centers (e.g. batch acquired points)
        X_train: (N, d) already-evaluated points (may be empty)
        threshold: scalar minimal allowed distance
        mode: "filter" or "greedy"
            - "filter": remove any center whose distance to X_train < threshold
                and also remove duplicates among centers (fast).
            - "greedy": sequentially select centers (ordered by sort_scores if given,
                otherwise in given order) but only accept center if it's >=threshold
                away from X_train and >=threshold away from already selected centers.
        sort_scores: optional (M,) array to sort centers (e.g. acquisition score);
                     higher score first if mode is "greedy".
        tol_dedup: tolerance for rounding dedup within new centers.

    Returns:
        selected_centers: (K, d) numpy array of selected centers
        selected_idx: indices into `centers` of kept centers
    """
    centers = np.asarray(centers)
    if centers.ndim != 2:
        raise ValueError("centers must be shape (M, d)")
    M, d = centers.shape

    # Deduplicate centers by rounding (cheap)
    if M > 1:
        uniq_idx = np.unique(np.round(centers / tol_dedup).astype(np.int64), axis=0, return_index=True)[1]
        centers = centers[np.sort(uniq_idx)]
        # note: this changes M; could optionally preserve mapping but keep simple

    # Build KD-tree for X_train if not empty
    if X_train is not None and len(X_train) > 0:
        tree_train = cKDTree(X_train)
        d2train, _ = tree_train.query(centers, k=1)
    else:
        d2train = np.full(centers.shape[0], np.inf)

    if mode == "filter":
        mask = d2train >= threshold
        idxs = np.where(mask)[0]
        selected_centers = centers[idxs]
        return selected_centers, idxs

    elif mode == "greedy":
        # order centers
        if sort_scores is not None:
            order = np.argsort(-np.asarray(sort_scores))  # high to low
        else:
            order = np.arange(len(centers))
        selected = []
        selected_idx = []
        # build an incremental KDTree or just use brute force on selected (small set)
        for ii in order:
            if d2train[ii] < threshold:
                # too close to existing training data
                continue
            cand = centers[ii]
            if len(selected) == 0:
                selected.append(cand)
                selected_idx.append(ii)
                continue
            # check distance to already selected
            sel_arr = np.vstack(selected)
            # compute distances
            dd = np.linalg.norm(sel_arr - cand[None, :], axis=1)
            if np.all(dd >= threshold):
                selected.append(cand)
                selected_idx.append(ii)
        if len(selected) == 0:
            return np.zeros((0, d)), np.array([], dtype=int)
        return np.vstack(selected), np.array(selected_idx, dtype=int)

    else:
        raise ValueError("mode must be 'filter' or 'greedy'")
