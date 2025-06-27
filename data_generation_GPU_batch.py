"""
C_params_ALL --> batch --> C_params_batch, ..., C_params_batch
each will be tested on NoisyQubitSimulator()

This module implements a simulator for a noisy qubit in the rotating frame
the inputs are 
1. noise trajectories
2. control_pulses
3. Evoltion time T and window L  ...

Outputs are
E[O] for all O and for all rho_S
"""

# preample
import numpy as np
from functools import reduce
from itertools import product
from scipy.linalg import expm
from datetime import datetime
import time
import torch
import os
import pickle

from qubit_simulator__GPU import NoisyQubitSimulator

BATCH_SIZE = 40  # Adjust this depending on memory constraints
my_device = 'cuda'

###############################################################################
torch.cuda.empty_cache()

np.random.seed(seed = 42)
torch.manual_seed(42)
###############################################################################

# Simulate in batches
def run_simulation_in_batches(C_params_ALL, input_parameters):
    """
    This function handles the simulation logic for a batch of experiments.
    Replace the code below with the core logic for simulating qubit dynamics
    """    
    tot_expr = C_params_ALL.shape[0]
    results = []

    for i in range(0, tot_expr, BATCH_SIZE):
        """
        Loop over all batch 
        """
        # Extract the current batch of experiments
        C_params_batch = C_params_ALL[i:i+BATCH_SIZE]
        
        # Perform the simulation for this batch
        noisy_qubit = NoisyQubitSimulator(T=input_parameters["T"], 
                                          C_params=C_params_batch,
                                          C_shape="Gaussian", 
                                          MM= input_parameters["MM"], 
                                          K= input_parameters["K"],
                                          gamma= input_parameters["gamma"], 
                                          g= input_parameters["g"], 
                                          Omega = input_parameters['Omega'],
                                          MultiTimeMeas= False)
        result_batch = noisy_qubit.circuit_implement()
        
        # Accumulate results
        results.append(result_batch)
        print(f'On batch @ {i}-th expr. Time= {datetime.now().strftime("%H:%M:%S")}')

    # Concatenate all results
    final_results = torch.cat(results, dim=0)

    return final_results


def save_simulation_data(input_parameters, C_params_ALL, results, directory):
    """
    Save the input parameters, C_params_ALL, and the results to disk.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the input parameters
    with open(os.path.join(directory, 'input_parameters.pkl'), 'wb') as f:
        pickle.dump(input_parameters, f)
    
    # Save C_params_ALL tensor
    torch.save(C_params_ALL, os.path.join(directory, 'C_params_ALL.pt'))
    
    # Save the results tensor
    torch.save(results, os.path.join(directory, 'results.pt'))
    
    print(f"Data saved to {directory}")

# Loop over different input parameters
def run_simulations_for_multiple_params(C_params_ALL, input_param_list):
    """
    Run simulations for multiple sets of input parameters and save the results.
    """
    for idx, input_params in enumerate(input_param_list):
        print(f"Running simulation for input parameter set {idx+1}")
        
        # Run the simulation in batches
        results = run_simulation_in_batches(C_params_ALL, input_params)
        
        # Save the data
        g_gam=input_params["g"]/input_params["gamma"]
        omg_T = input_params["Omega"]*input_params["T"]
        save_simulation_data(input_params, C_params_ALL, results, directory= \
                              f"simulation_data/set_{idx+1}")
                            # f"simulation_data/set_{idx+1}___g_{g_gam}_Omg_{omg_T}")

###############################################################################
#           FOR SINGLE SET OF input_parameters 
## Placeholder for other inputs, parameters, etc.
#C_params_ALL = torch.randn(100, 4, 3, dtype=torch.complex64, device=my_device)  # Example C_params tensor
#input_parameters = {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**4,  "g": 0* 10* 10**5}

#tic = time.time()

## Run the simulation in batches
#results = run_simulation_in_batches(C_params_ALL, input_parameters)
#print(results)

#print("Simulation complete. Results shape:", results.shape)

#toc = time.time()
#print(f"Elapsed time:{toc-tic} seconds")

###############################################################################
# Example: Define multiple sets of input parameters
input_param_list = [
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 0* 10* 10**5,  "Omega": 0*10**6},

    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 1* 10* 10**5,  "Omega": 0*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 2* 10* 10**5,  "Omega": 0*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 5* 10* 10**5,  "Omega": 0*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 10* 10* 10**5, "Omega": 0*10**6},

    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 1* 10* 10**5,  "Omega": 0.5*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 2* 10* 10**5,  "Omega": 0.5*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 5* 10* 10**5,  "Omega": 0.5*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 10* 10* 10**5, "Omega": 0.5*10**6},

    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 1* 10* 10**5,  "Omega": 1*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 2* 10* 10**5,  "Omega": 1*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 5* 10* 10**5,  "Omega": 1*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 10* 10* 10**5, "Omega": 1*10**6},

    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 1* 10* 10**5,  "Omega": 2*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 2* 10* 10**5,  "Omega": 2*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 5* 10* 10**5,  "Omega": 2*10**6},
    {"T": 10**-6, "MM": 1000, "K": 1024, "gamma": 10**5, "g": 10* 10* 10**5, "Omega": 2*10**6},
    # Add more parameter sets as needed
]

# Example C_params_ALL tensor (modify this with actual data)
C_params_ALL = torch.randn(1000, 4, 3, dtype=torch.complex64, device= my_device)  # Example tensor

X_pulses_ALL = torch.randn(1000, 4, 1,  dtype=torch.complex64)  # make a rand control tensor only along X axis
C_params_ALL = torch.cat((X_pulses_ALL, torch.zeros(1000,4,2)), dim=2).\
                to(torch.complex64).to(device=my_device)          # Make a X-ctrl tensor that shape [C, L=4, 3]]

# Run simulations for multiple sets of input parameters
run_simulations_for_multiple_params(C_params_ALL, input_param_list)


if __name__ == '__main__':
    # load input params from set_1
    with open('simulation_data/set_1/input_parameters.pkl', 'rb') as f:
        input_params = pickle.load(f)

    # load the C_params_ALL tensor
    C_params_ALL = torch.load('simulation_data/set_1/C_params_ALL.pt',weights_only=True)

    # load the result tensor
    results = torch.load('simulation_data/set_1/results.pt',weights_only=True)

    print("Loaded data:", input_params, C_params_ALL.shape, results.shape)
                              