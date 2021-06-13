# -*- coding: utf-8 -*-
"""
Created on Tue Sept 22 14:44:02 2020

@author:    Jonas Hartmann @ De Renzis group @ EMBL Heidelberg

@descript:  A collection of generally helpful functions.
"""

import numpy as np
import scipy.stats as stats


def running_mean(t, y, window=1, min_samples=9):
    """Compute running mean of values y over time t.

    Note that input arrays with ndim>1 are flattened.

    Parameters
    ----------
    t : nd array
        Time points over which the windows are applied.
    y : nd array
        Values to be averaged at each time point.
        Must have the same size as t.
    window : int, default 1
        The size of the running window is `2*window + 1`.
    min_samples : int, default 9
        Minimum number of samples within a time window.
        The result from time points with fewer samples
        is not returned.

    Returns
    -------
    tpoints : 1d array
        Time points for which the running mean has been computed.
    ymeans : 1d array
        Mean values at the given time points.
        Has the same size as tpoints.
    """

    # Flatten input and remove nans
    t = t.flatten()
    y = y.flatten()
    t = t[~np.isnan(t)]
    y = y[~np.isnan(y)]
    
    # Confirm equal length
    if not len(t)==len(y):
        raise Exception("t and y must hold the same number of values!")
    
    # Get all unique time points
    tpoints = np.unique(t)
    
    # Find data within `window` of tpoints & compute mean
    ymeans   = np.zeros_like(tpoints, dtype=np.float)
    nsamples = np.zeros_like(tpoints)
    for i,tp in enumerate(tpoints):
        mask = ((tp-window) <= t) & (t <= (tp+window))
        nsamples[i] = np.sum(mask)
        if nsamples[i] > 0:
            ymeans[i] = np.mean(y[mask])
        else:
            ymeans[i] = np.nan
    
    # Remove points with too few samples
    tpoints = tpoints[nsamples >= min_samples]
    ymeans  = ymeans[nsamples >= min_samples]
    
    return tpoints, ymeans


def running_CI(t, y, ci_type="single", window=1, min_samples=9):
    """Compute a running confidence interval for values y over time t.

    Note that input arrays with ndim>1 are flattened.

    Parameters
    ----------
    t : nd array
        Time points over which the windows are applied.
    y : nd array
        Values over which the CI is calculated at each time point.
        Must have the same size as t.
    ci_type : str, default "single"
        Type of confidence interval used:
        - "mean": CI of the mean
        - "single": CI of a single draw (default)
        - "stdev": Standard deviation (not CI)
    window : int, default 1
        The size of the running window is `2*window + 1`.
    min_samples : int, default 9
        Minimum number of samples within a time window.
        The result from time points with fewer samples
        is not returned.

    Returns
    -------
    tpoints : 1d array
        Time points for which the running mean has been computed.
    ymeans : 1d array
        Mean values at the given time points.
        Has the same size as tpoints.
    """

    # Flatten input and remove nans
    t = t.flatten()
    y = y.flatten()
    t = t[~np.isnan(t)]
    y = y[~np.isnan(y)]
    
    # Confirm equal length
    if not len(t)==len(y):
        raise Exception("t and y must hold the same number of values!")
        
    # Get all unique time points
    tpoints = np.unique(t)
    
    # Find data within `window` of tpoints & compute CI
    yCI      = np.zeros((len(tpoints), 2), dtype=np.float)
    nsamples = np.zeros_like(tpoints)
    for i,tp in enumerate(tpoints):
        mask = ((tp-window) <= t) & (t <= (tp+window))
        nsamples[i] = np.sum(mask)
        ym = y[mask]
        
        if nsamples[i] >= min_samples:
            
            # Catch exceptional case where all values are the same
            if np.all(ym==ym.mean()):
                yCI[i,:] = (ym[0], ym[0])
            
            elif ci_type=="single":
                yCI[i,:] = stats.norm.interval(0.95, loc=np.mean(ym), scale=np.std(ym))
                
            if ci_type=="mean":
                yCI[i,:] = stats.t.interval(0.95, len(ym)-1, loc=np.mean(ym), scale=stats.sem(ym))
                
            elif ci_type=="stdev":
                yCI[i,:] = (np.mean(ym)-np.std(ym), np.mean(ym)+np.std(ym))
                
        else:
            yCI[i] = np.nan
    
    # Remove points with too few samples
    tpoints = tpoints[nsamples >= min_samples]
    yCI     = yCI[nsamples >= min_samples]
    
    return tpoints, yCI

