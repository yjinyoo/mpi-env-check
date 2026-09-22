# mpi-env-check

Assert at the top of a parallel job that the environment is the one the script
assumes, and exit if it is not.

```python
from mpi_env_check import require_parallel_env
env = require_parallel_env(expect_ranks=8, max_threads=1)
```

```
python mpi_env_check.py     # report the environment, exit 0 or 1
```

## The failure it catches

A parallel job that reports success and produces wrong or worthless work.

Four of them, each a different mechanism, each invisible at the time:

- The launcher could not attach the MPI library, so 32 ranks each became rank 0
  of a world of size 1. Every `world.rank == 0` guard passed on all 32. They
  duplicated one calculation serially and wrote the same files at once.
- A write helper was a collective call, so only rank 0 produced the file an
  external binary was about to read. The other 31 pointed at a path that did not
  exist. Invisible while every rank was its own world.
- Nine runs exited 0 and five of their archives were unreadable, because all 32
  ranks wrote the same path.
- Thread count was unset, so each rank spawned 33 BLAS threads. 264 threads
  fought over 16 cores and a step took four times what it should. `top` showed
  every rank at 94 percent.

The shape is the same every time. The launcher, the threading library or the MPI
world is not what the script assumed, nothing raises, and the only symptom is a
number nobody was looking at. Each post-mortem ended with the same instruction,
which nobody put into code, which is why there were four.

So this is a gate, not a report. It exits.

## What it checks

`require_parallel_env` compares the live environment against what you declare:

| declared | checked against |
| --- | --- |
| `expect_ranks` | the actual MPI world size, or `SLURM_NTASKS` if you leave it out |
| `max_threads` | `OMP_NUM_THREADS` (must be set), `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` (if set), and the threads this process actually holds |
| (nothing) | the ranks the launcher put on this node, against its physical cores |

A world size of 1 when you asked for 8 is the first failure above, and it is the
one worth gating hardest: it is the only one where every rank believes it is in
charge.

Threads are checked twice because each check misses something. Measured under
`mpirun` with the variables unset:

- conda-forge numpy (OpenMP BLAS) held 3 threads at the top of the job and 18
  after one matrix product. The live count (from `/proc/self/task`) sees only
  threads that already exist, so only the variable check catches this.
- pip numpy, as used with GPAW, held 18 threads from import. GPAW sets
  `OMP_NUM_THREADS=1` when it is imported, so if numpy was imported first the
  variable reads 1 while 18 threads already run. Only the live count catches
  this.

The live count needs `/proc`, so on macOS and Windows it is skipped and the
start-up line says so. `max_threads=0` turns both off.

Ranks per node come from `OMPI_COMM_WORLD_LOCAL_SIZE` (Open MPI),
`MPI_LOCALNRANKS` (MPICH, Intel MPI) or Slurm's per-node task list. On a
multi-node job where none of these is set, that check is skipped rather than
guessed.

`describe_env()` returns the same facts without exiting, for logging at the start
of a run. Print it. A run whose log does not record its own world size cannot be
diagnosed later, and that is how the same accident repeats.

## The write side

```python
from mpi_env_check import rank0_write
rank0_write(path, lambda p: np.savez_compressed(p, **arrays))
```

Runs on rank 0, barriers, and raises the same exception on every rank. The last
part matters: a naive rank-0 write that throws leaves the other ranks waiting at
a barrier that never comes, and the job hangs until the wall clock kills it, with
no error anywhere.

## Which MPI

It finds the world from `gpaw.mpi` or `mpi4py`, whichever is importable, and
falls back to a serial stand-in so a script written for a cluster still runs on
one machine. The stand-in is a world of size 1 whose source is reported as
`serial`, so a log line tells "no MPI at all" apart from "an MPI world of size
1", which is the first failure in the list.

## Tests

```
python test_mpi_env_check.py
```

No MPI needed: the world, the thread count and the core count are replaced.

## Requirements

Python 3.9+, standard library only. An MPI library is detected if present and
not required.

MIT licensed.
