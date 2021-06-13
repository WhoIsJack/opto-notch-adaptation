# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:29:44 2021

@author:    Jonas Hartmann @ De Renzis group @ EMBL Heidelberg

@descript:  Loading & preparing data for model fitting.
"""

from collections import defaultdict
import numpy as np


def load_data(dpath):
    """Load relevant data from compiled data file. 

    Note: This file is generated in `ANA - sim spot detection.ipynb`.
    """

    data = defaultdict(lambda : {'times' : [], 'counts' : [], 'ints' : []})

    with open(dpath, 'r') as infile:
        for line in infile.readlines():
            
            line = line.strip().split('\t')
            numarr = np.array([float(i) for i in line[3:]])
            
            if line[0] in ['continuous', 'pulsatile']:
                
                if line[2] == 'TIMES:':
                    data[line[0]]['times'].append(numarr)

                if line[2] == 'COUNTS:':
                    data[line[0]]['counts'].append(numarr)
                    
                if line[2] == 'TOTALINTS:':
                    data[line[0]]['ints'].append(numarr)

    return data


def truncate_exp(data, exp_end_times={'continuous' : 40.0, 'pulsatile' : 32.0}):
    """Truncate data to a uniform experiment end time."""

    for exp in data:
        for t, times in enumerate(data[exp]['times']):
            times_mask  = times <= exp_end_times[exp]
            data[exp]['times'][t] = times[times_mask]
            data[exp]['counts'][t] = data[exp]['counts'][t][times_mask]
            data[exp]['ints'][t] = data[exp]['ints'][t][times_mask]

    return data


def rescale_time(data,
    sim_end_times = {'continuous' : 40.0, 'pulsatile' : 32.0},
    exp_end_times = {'continuous' : 40.0, 'pulsatile' : 32.0}):
    """Rescale time to match experiment and model time scales.

    Note: This is no longer required in the current version.
    """

    for exp in data:
        for t, times in enumerate(data[exp]['times']):
            times_scaled = times / exp_end_times[exp] * sim_end_times[exp]
            data[exp]['times'][t] = times_scaled

    return data


def rescale_intensity(data, max_totalint=300):
    """Rescale total intensities to a more reasonable scale.

    Note: This is for computational convenience (to avoid very large MSE values)
    but changes nothing in the fitting, as intensity values are on an arbitrary
    scale to begin with.
    """

    for exp in data:
            
        for t, ints in enumerate(data[exp]['ints']):
            ints_scaled = ints / max_totalint
            data[exp]['ints'][t] = ints_scaled

    return data

