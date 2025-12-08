# Paper title

AL/BO approaches in the context of experimental settings:

- fixed budget, 100 pts (may not be enough for high-D datasets)
- cycle batches & acquisition learning mix/protocols
- optmization vs representation (balancing exploration and exploitation)

Motivations:

- ...

## Assessing a AL experiment

1. Convergence metrics:

    - **Best value over time (screened)**: simple plot of the best _screened_ values across cycles (or repetition)
    - **Simple regret**: simple regret defined as true optimum (required from GT) minus best screened at a specific cycle. Shows distance to best value. Log scale emphasizes early progress and shows convergence rate.
    - **Time to threshold of the optimum**: counts how many cycles, which means how many points, are needed to read a X% of the optimum (required from GT or it uses the last value screened by the individual experiment).

2. Efficiency metrics:

    - **Sample efficiency**: plot best value as a function of sampled points (not cycles). The different batched are taken into account. It reveals the sample efficiency of each methods.
    Some further interpretations:
        - Steeper slope, means faster improvement early
        - Reach high values with fewer samples, means that it does not waste screened points.
        - Consisten growth, means reliable progress.
    - **Batch diversity**: computes the Pairwise distances between observations in n-dimensional space for each batch of points collected in a cycle.

3. Performance metrics:

    - **RMSE (Root Mean Squared Error)**: calculates the square root of the average of squared errors. Highly sensitive to outliers because squaring large errors makes them even larger.
    - **MAE (Mean Absolute Error)**: calculates the average of the absolute errors. Less sensitive to outliers because it does not square the errors (treats all errors equally).
    - **PICP95%**: "Prediction Interval Coverage Probability at 95%". Measures the percentage of true values that fall within the 95% prediction intervals of the model.\
    `PICP = (Number of points inside interval) / (Total points) × 100%`\
    Ideal value is equal to the N% interval, as for a well calibrated model N% of the prediction should capture the true value.
        - Above N%: the model is over-conservative (uncertainty overestimated)
        - Below N%: the model is over-confident (uncertainty underestimated)
    - **MPIW95%**: "Mean Prediction Interval Width at 95%". Measures the average width of the 95% prediction intervals. Lower is better, as narrower intervals mean more confident, precise predictions. It should decrease over time as you collect more data, but must balance with PICP.

4. Visualization tools:

    - **Acquisition source distribution**: visualization of the protocol as a stacked rectangular area plot, showing the composition over time.

## 3D study case

Motivations:

- simple/common set up for experiments, like explorative/simple material property optimization, phase diagram optimization
- set up a baseline for comparisons

### 'Dummy' AL approach

Motivation: in experimental settings screening only 1 point per cycle might not efficient if the experimental validation is the bottleneck. Using a batch/protocol approach we would like to get a better understanding of the system, not only focused on Exploration or Exploitation but a mix of the two.

1. No batches, but single point acquisition, favoring exploration of exploitation; all compared against a random selection of points.

    - Set the starting point from which we would like to optimize by using a batch/protocol approach.

### Balance between exploration and exploitation, and experiment tuning

Motivation: showcasing how batch AL can be employed to have a better efficiency for the AL experiments, with regards to wet-experimental settings as well.

1. Purely exploitative experimnet setups

    - Performance in finding (screening) the maximum: lower batch number (high cycles) provides a faster way of screening the best points, but usually penalizes the overall RMSE/MAE of the predicted landscape
    - ranking 10-10 -> 100-1 in terms of 'speed' of screening the maximum (faster regret and efficiency)
    - ranking 20-5/50-2 -> 10-10/100-1 in terms of RMSE/MAE consistenly lower after a buffer amount of cycles/points (around 40 tot. points screened points)
    - highest batch per cycles (10-10) worst performances overall, due to lowest sample diversity (Pair-wise distance points in batch)
    - 20-5 (and 50-2 secondly) has the highest batches diversity

2. Purely explorative experiment setups

    - May never screen 99%/95% of the target value (overall optimum). Up to 90% the behaviour is very herratic (with 10-10 beingh better and 20-5 almost needing the entire 100 pts). On average the batch diversity is high or at least comparable with the top diverse exploitation methods.
    - RMSE/MAE in line with expectations: overall slightly better than exploitative cases.Model confidence in general higher tham the exploitative methods.

3. Protocol acquisition experimental setups

    - Using unit of 20pts per stage (idk how to justify it).
    - The idea is that I would like to improve the lowest performing single acquisition experiments benefitting of the lower amounts of cycles
    - Tested on the worse case, 10-10, keeping batch pure and introducing a staged protocol, where exploration and exploitation interchange:
    - Only tested 80% and 50% (in EI cycle count) as introducing to many MI cycles would for sure destroy the optimization performance.
    - Both cases show a decrese in 'points to performance thresholds' speed compared to the purely EI experiment.
    - RMSE/MAE not improved enough compared to the other experimental setups

4. Mixed acquisitions experiment setups

    - Taken the worst case (10-10) it is possible to improve the 'points to performance thresholds' by adding an explorative part to the point batches:

        - 8 EI, 2 MI gives an improvemen
        - 50% or 20% split (in EI count) gives a notably worse performance, while having a higher and steady batch diversity.

    - Other cases generally benefits from the batch split:

        - ...

### Noise effect on AL performances

Motivation: building from the results and findings of the previous sections we showcase on some relevant experiment setups the effect of noise (often affecting the experimental validation) on the screened points.

## Needle in a haystack: extreme optimization function

Motivation: showcase how in the case of an extreme function to optimize the performances changes.

## High-dimensional optimization functions

Motivation: showcasing how increasing the dimensionality has an impact on the performances and might change the previous understanding of batch/protocol AL.
