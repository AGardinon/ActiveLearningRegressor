#!

import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Union, List, Tuple


def sample_landscape(X_landscape: np.ndarray, n_points: int, sampling_mode: str='fps', **kwargs) -> np.ndarray:
    """
    Always returns an index from the X_landscape array
    """

    sampling_mode_dict = {
        'fps' : fps,
        'voronoi' : voronoi,
        'random' : rnd
    }

    return sampling_mode_dict[sampling_mode](X=X_landscape, n_points=n_points, **kwargs)


def rnd(X: np.ndarray, n_points: int) -> np.ndarray:
    """
    Sample n new points in a random fashion from the grid space.
    
    Parameters:
    - X: ...
    - n_points: Number of new points to sample
    
    Returns:
    - new_indices: Indices of the new points on the grid
    """
    
    indices_pool = np.arange(len(X))

    if not isinstance(indices_pool, list):
        indices_pool = list(indices_pool)

    return random.sample(indices_pool, n_points)


def fps(X: np.ndarray, 
        n_points: int, 
        start_idx: int=None, 
        return_distD: bool=False) -> Union[List[int], Tuple[List[int],np.ndarray]]:

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
    

def voronoi(X, n_points, mode='MiniBatchKMeans'):
    
    cluster_mode = {
        'KMeans' : KMeans(init="k-means++", n_clusters=n_points),
        'MiniBatchKMeans' : MiniBatchKMeans(init="k-means++", n_clusters=n_points),
    }

    kmeans = cluster_mode[mode].fit(X)

    # select the k samples closest to their respective cluster centroid
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    return closest