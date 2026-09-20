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
| `max_threads` | the thread-count environment variables, and the cores available |

A world size of 1 when you asked for 8 is the first failure above, and it is the
one worth gating hardest: it is the only one where every rank believes it is in
charge.

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
one machine. The stand-in reports `serial` rather than pretending to be a world
of size 1, because that pretence is the first failure in the list.

## Requirements

Python 3.9+, standard library only. An MPI library is detected if present and
not required.

MIT licensed.
