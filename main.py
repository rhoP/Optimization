# main.py
import json
import importlib
import jax.numpy as jnp
from jax import grad
import numpy as np
from mpi4py import MPI
from core.utils import load_config


def evaluate_function_and_gradient(func, x):
    x = jnp.array(x)
    y = func(x)
    gradient = grad(func)(x)
    return y, gradient


# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Load configuration
config_path = 'config/config.json'
config = load_config(config_path)

# Import the specified objective function
objective_module = importlib.import_module(f"objectives.{config['objective_function']}")
objective_function = getattr(objective_module, config['objective_function'])

# Import the specified optimization algorithm
algorithm_module = importlib.import_module(f"algorithms.{config['algorithm']}")
algorithm_function = getattr(algorithm_module, config['algorithm'])

# Define the distributed ADMM execution
if config['algorithm'] == 'distributed_admm':
    consensus_func_module = importlib.import_module(f"objectives.{config['consensus_function']}")
    consensus_function = getattr(consensus_func_module, config['consensus_function'])
    initial_x = np.array(config['initial_x'], dtype=np.float32)
    max_iters = config['max_iters']
    rho = config['rho']

    # Determine local data range
    data_size = len(initial_x)
    chunk_size = data_size // comm.Get_size()
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < comm.Get_size() - 1 else data_size

    # Run distributed ADMM
    x_local, z, u = algorithm_function(
        objective_func=lambda x: objective_function(x, start, end),
        consensus_func=consensus_function,
        x_init=initial_x[start:end],
        rho=rho,
        max_iters=max_iters
    )

    # Gather results
    result = comm.gather(x_local, root=0)
    if rank == 0:
        result = np.concatenate(result)
        print("Optimization Result:", result)
else:
    # Run other optimization algorithms
    result = algorithm_function(
        func=objective_function,
        initial_x=config['initial_x'],
        learning_rate=config['learning_rate'],
        max_iters=config['max_iters']
    )
    if rank == 0:
        print("Optimization Result:", result)
