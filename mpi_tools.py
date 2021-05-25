import os
import sys
import torch
import subprocess
from mpi4py import MPI


def num_procs() -> int:
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def proc_id() -> int:
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def mpi_fork(n: int, bind_to_core: bool = False) -> None:
    """
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name`_.
    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n: Number of process to split into.
        bind_to_core: Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", IN_MPI="1")
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using more than its fair share of CPU resources.
    """
    print(f'Proc {proc_id()}: Reporting original number of Torch threads as {torch.get_num_threads()}.', flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    print(f'Proc {proc_id()}: Reporting new number of Torch threads as {torch.get_num_threads()}.', flush=True)
