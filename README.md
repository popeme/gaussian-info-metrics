# gaussian-info-metrics
A library to calculate local, multivariate information theory metrics on Gaussian-distributed variables.

# what's here?
- `local_entropy`: calculates the shannon "surprisal" for one state of a one-dimensional variable.
- `local_entropy_2d`: caclulates the shannon "surprisal" for one state of a two-dimensional variable.
- `local_gaussian_joint_entropy`: calculates the joint information content of every state (in the input array) for an N-dimensional variable.
- `gaussian_mi`: calculates the mutual information between two input time-series. A closed form conversion from Pearson correlation
- `gaussian_pmi`: calculates the local mutual information for one state of a two-dimensional variable.
- `local_edge_timeseries`: calculates the local mutual information between every pair of input variables at every point in time
- `local_total_correlation`: calculates the total correlation for every state in an N-dimensional time series
- `local_dual_total_correlation`: calculates the dual total correlation for every state in an N-dimensional time series
- `local_o_information`: calculates the o information for every state in an N-dimensional time series
- `local_s_information`: calculates the s information for every state in an N-dimensional time series

# How to Use
1. Download the library
2. Call `info_library.function_name_here(X)` where X is your time-series

# Important Notes
- The library assumes that all time series are Gaussian distributed! Do not use for variables from any other distributions!
- All inputs should always be variables x time
- The library will accept an optional second input time series. If your data is a smaller sample of a much larger dataset, (for example, a single subject in a dataset of many subjects), we recommend always inputting the full dataset as the second input time series. The covariance matrix that defines the Gaussian will then be calculated from the second time series rather than the first. This allows for more samples, and so a better estimation of the true joint probabilities. Failure to input the full dataset can result in systematic underestimation of quantities like the O-information (i.e. you will see that your data is more synergy-dominated than it actually is). 
