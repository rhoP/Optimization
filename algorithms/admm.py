# algorithms/distributed_admm.py
from mpi4py import MPI
import numpy as np


def distributed_admm(objective_func, consensus_func, x_init, rho=1.0, max_iters=100):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    x_local = np.array(x_init, dtype=np.float32)
    z = np.zeros_like(x_local)
    u = np.zeros_like(x_local)

    def update_local_x(x_local, z, u, rho):
        return x_local - (u + rho * (x_local - z))

    def update_global_z(x_local, u, rho, size):
        x_avg = np.mean(x_local + u / rho, axis=0)
        return consensus_func(x_avg / size)

    for iteration in range(max_iters):
        x_local = update_local_x(x_local, z, u, rho)

        x_local_all = np.array(comm.allgather(x_local))
        z = update_global_z(x_local_all, u, rho, size)

        u += x_local - z

        if rank == 0:
            print(f"Iteration {iteration}: x_local = {x_local}, z = {z}, u = {u}")

    return x_local, z, u


# Example consensus function (could be sum, average, etc.)
def consensus_func(x):
    return x
