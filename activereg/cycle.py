#!

from activereg.sampling import sample_landscape
from activereg.acquisition import AcquisitionFunction, highest_landscape_selection


def active_learning_cycle(
        X_candidates,
        y_candidates,
        y_best,
        model, 
        acquisition_parameters,
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
    acqui_fun = AcquisitionFunction(y_best=y_best, **acquisition_parameters)
    y_pred, landscape = acqui_fun.landscape_acquisition(X_candidates=X_candidates, ml_model=model)
    
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