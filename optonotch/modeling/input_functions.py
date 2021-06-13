# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:03:03 2021

@author:    Jonas Hartmann @ De Renzis group @ EMBL Heidelberg

@descript:  Functions to simulate model inputs (nuclear NICD).
"""

# Imports
import os
import dill
import numpy as np


# Load the fitted import-export functions
try:
    fitted_path = __loader__.get_filename()
    fitted_path = os.path.split(fitted_path)[0]
    fitted_path = os.path.abspath(fitted_path)

    with open(os.path.join(fitted_path, 'fitted_import_func_sympy_bc.pkl'), 'rb') as infile:
        on_model = dill.load(infile)
    with open(os.path.join(fitted_path, 'fitted_export_func_sympy_bc.pkl'), 'rb') as infile:
        off_model = dill.load(infile)

except FileNotFoundError:
    print("Error: Fitted import/export function file not found. " +
          "Something may be wrong with the path.")
    raise


# Define the various input functions
def input_Ferrell(t):
    """Mimicking the inputs used in Ferrell, 2016"""
    
    steps  = np.array([20, 50, 60, 70, 80, 90, np.inf])
    values = np.array([0.0, 0.2, 0.4, 0.6, 0.4, 0.2, 0.0])
    
    index_t = np.searchsorted(steps, t)
    value_t = values[index_t]
    
    return value_t


def input_continuous_linear(t):
    """Simple linear approximation to continuous activation."""

    sim_end_time = 32.0
    exp_end_time = 32.0
    
    exp_import_time = 10.0
    sim_import_time = exp_import_time / exp_end_time * sim_end_time
    
    if t <= sim_import_time:
        return t / sim_import_time
    else:
        return 1.0


def input_pulsatile_linear(t):
    """Simple linear approximation to pulsatile activation."""
    
    sim_end_time = 32.0
    exp_end_time = 32.0  
    exp2sim = lambda exp : exp / exp_end_time * sim_end_time
    
    exp_import_time = 10.0
    sim_import_time = exp2sim(exp_import_time)
    
    exp_export_time = 10.0
    sim_export_time = exp2sim(exp_export_time)
    
    if t <= exp2sim(5.0):
        return np.max([0.001, t / sim_import_time])
    
    elif t <= exp2sim(14.0):
        start_t = exp2sim(5.0)
        start_n = input_pulsatile_linear(start_t)
        return np.max([0.001, start_n - ((t - start_t) / sim_export_time)])
                   
    elif t <= exp2sim(19.0):
        start_t = exp2sim(14.0)
        start_n = input_pulsatile_linear(start_t)
        return np.max([0.001, start_n + ((t - start_t) / sim_import_time)])
                   
    elif t <= exp2sim(28.0):
        start_t = exp2sim(19.0)
        start_n = input_pulsatile_linear(start_t)
        return np.max([0.001, start_n - ((t - start_t) / sim_export_time)])
                   
    elif t <= exp2sim(32.0):
        start_t = exp2sim(28.0)
        start_n = input_pulsatile_linear(start_t)
        return np.max([0.001, start_n + ((t - start_t) / sim_import_time)])
                   
    else:
        return input_pulsatile_linear(exp2sim(32.0))


def input_continuous_model(t, N0=0.0):
    """Empirically fitted import-export model for continuous activation."""

    return on_model(N0, t)


def input_pulsatile_model(t, N0=0.0,
                          steps={'off_step1':4.5,  'on_step2':14.5, 
                                 'off_step2':19.0, 'on_step3':29.0, 
                                 'end_step':32.0}):
    """Empirically fitted import-export model for pulsatile activation."""
    
    if t <= steps['off_step1']:
        return on_model(N0, t)
    
    elif t <= steps['on_step2']:
        pN0 = input_pulsatile_model(steps['off_step1'], N0=N0, steps=steps)
        return off_model(pN0, t-steps['off_step1'])
                   
    elif t <= steps['off_step2']:
        pN0 = input_pulsatile_model(steps['on_step2'], N0=N0, steps=steps)
        return on_model(pN0, t-steps['on_step2'])
                   
    elif t <= steps['on_step3']:
        pN0 = input_pulsatile_model(steps['off_step2'], N0=N0, steps=steps)
        return off_model(pN0, t-steps['off_step2'])
                   
    elif t <= steps['end_step']:
        pN0 = input_pulsatile_model(steps['on_step3'], N0=N0, steps=steps)
        return on_model(pN0, t-steps['on_step3'])
                   
    else:
        return input_pulsatile_model(steps['end_step'], N0=N0, steps=steps)