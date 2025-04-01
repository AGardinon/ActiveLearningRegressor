#!

from activereg.sampling import sample_landscape
from activereg.acquisition import landscape_acquisition, highest_landscape_selection


def active_learning_cycle(
        X_candidates,
        y_candidates,
        model, 
        acquisition_mode, 
        percentile,
        n_batch,
        sampling_mode
        ):
    """
    Performs the active learning regression cycle:
    1.  compute the navigation landscape
    2.  select top-portion of the landscape
    3.  sample new points from the selected portion
    """

    # 1.
    y_pred, landscape = landscape_acquisition(X_candidates=X_candidates, 
                                              ml_model=model, 
                                              acquisition_mode=acquisition_mode)
    
    # 2.
    acq_landscape_ndx = highest_landscape_selection(landscape=landscape, 
                                                    percentile=percentile)
    X_acq_landscape = X_candidates[acq_landscape_ndx]

    # 3.
    sampled_hls_idx = sample_landscape(X_landscape=X_acq_landscape,
                                       n_points=n_batch,
                                       sampling_mode=sampling_mode)
    sampled_new_idx = acq_landscape_ndx[sampled_hls_idx]
    
    # 4.
    X_next, y_next = X_candidates[sampled_new_idx], y_candidates[sampled_new_idx]

    return X_next, y_next, y_pred, landscape, X_acq_landscape, sampled_new_idx