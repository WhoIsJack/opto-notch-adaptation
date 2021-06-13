# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:15:39 2021

@author:    Jonas Hartmann @ De Renzis group @ EMBL Heidelberg

@descript:  Loss functions for model fitting.
"""

import warnings

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import odeint


def mse_loss(params, model, ini, t, 
             input_signal, data_times, 
             data_values, scale_factor,
             stabilize=False):
    """Mean Square Error loss function. 
    
    This function integrates `model` with `params` and `input_signal` using `odeint`, 
    then interpolates simulation results for all `data_times` and rescales the model 
    output to fit the data scale by multiplying with `scale factor`, and finally 
    computes the MSE between rescaled simulation results and `data_values`.

    Parameters
    ----------
    params : iterable of numbers
        Parameters for the ODE model.
    model : callable
        ODE model function.
    ini : iterable of numbers
        Initial values of ODE variables.
    t : iterable of numbers
        Time steps to use in `odeint` integration.
    input_signal : callable
        Function that returns the value of the NICD
        input given an experiment time t.
    data_times : iterable
        Time of each measured data point.
    data_values : iterable
        Value of each measured data point.
    scale_factor : float
        Factor by which model outputs are multiplied
        to match the scale of the data.
    stabilize : bool, default False
        If True, cases where odeint fails are caught
        and return `np.inf`. This is because some 
        models (i.e. the Hill variants) can produce
        errors with some parameters, which should 
        not stop the optimization from proceding.

    Returns
    -------
    MSE : float
        Total mean squared error between the rescaled
        simulations output and all measured data points.
    """
    
    # Run numerical solver (normal case)
    if not stabilize:
        solution = odeint(model, ini, t, args=(params, input_signal))
        output_signal = solution[:,0]

    # Run numberical solver (with error catching)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(action='error')
                solution = odeint(model, ini, t, args=(params, input_signal))
                output_signal = solution[:,0]
        except Exception:
            return np.inf

    # Interpolate signal at data times
    interp_signal = np.interp(data_times, t, output_signal)
    
    # Rescale to real data
    scaled_signal = interp_signal * scale_factor
    
    # Compute mean square error
    MSE = np.mean((data_values - scaled_signal)**2.0)
    
    # Done
    return MSE