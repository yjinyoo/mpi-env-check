"""Assert that the parallel environment is the one the script assumes.

WHY THIS EXISTS
---------------
Four times now a parallel job has produced wrong or worthless work while
reporting success, and every time the post-mortem ended with the same
instruction, which nobody then put into code:

  2026-08-27  `srun` could not attach conda's OpenMPI, so 32 ranks each became
              rank 0 of a world of size 1. Every `world.rank == 0` guard passed
              on all 32, they duplicated the same calculation serially and wrote
              the same files at once. Lesson recorded: "print world.size".
  2026-08-28  `ase.io.write` is a parallel function, so only rank 0 wrote the
              coordinates an external binary was about to read; the other 31
              ranks pointed at a file that did not exist. Invisible while every
              rank was its own world.
  2026-09-03  Nine DFT campaigns finished with exit code 0 and five of their
              archives were `BadZipFile`, because all 32 ranks wrote the same
              path. Lesson recorded: "print world.size at the start, open your
              own output at the end".
  2026-09-06  Each rank spawned 33 BLAS threads because OMP_NUM_THREADS was
              unset, so 264 threads fought over 16 cores and one SCF step took
              four times what it should. `top` still showed every rank at 94 %.

The common shape is that the launcher, the threading library or the MPI world
is not what the script assumes, nothing raises, and the only visible symptom is
a number nobody was looking at. So this is a gate, not a report: it exits.

USE
---
    from mpi_env_check import require_parallel_env
    env = require_parallel_env(expect_ranks=8, max_threads=1)

at the very top of the job, before the calculator is built. Under Slurm the
rank count comes from `SLURM_NTASKS` if `expect_ranks` is not given.

For the write side:

    from mpi_env_check import rank0_write
    rank0_write(path, lambda p: np.savez_compressed(p, **arrays))

which runs on rank 0, barriers, and raises the same exception everywhere rather
than leaving the other ranks waiting at a barrier that never comes.

    python mpi_env_check.py            # report the environment and exit 0/1
"""
from __future__ import annotations

import os
import re
import sys

__all__ = ["describe_env", "require_parallel_env", "rank0_write",
           "physical_cores", "ranks_on_this_node"]

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

# Threads a healthy rank holds beyond its compute threads: the interpreter and
# the MPI library keep their own. Three measured with mpi4py on Open MPI and
# OMP_NUM_THREADS=1 (WSL, 2026-09-22).
RUNTIME_THREADS = 3


def _world():
    """The MPI world, whatever library provided it, or a serial stand-in."""
    try:
        from gpaw.mpi import world
        return world.rank, world.size, "gpaw.mpi"
    except Exception:
        pass
    try:
        from mpi4py import MPI
        c = MPI.COMM_WORLD
        return c.Get_rank(), c.Get_size(), "mpi4py"
    except Exception:
        return 0, 1, "serial"


def threads_in_this_process() -> int:
    """Threads the OS says this process holds. The number that caught 09-06.

    Not the same as the environment variable: a library that ignored the
    variable, or was configured before it was set, still shows up here.
    """
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return -1                                  # not Linux, unknowable here


def physical_cores() -> int:
    """Physical cores, not hyperthreads.

    `os.cpu_count()` and `nproc` report hyperthreads, and Open MPI counts a slot
    per physical core, so using cpu_count as the rank count is refused with
    "not enough slots available".
    """
    try:
        ids = set()
        with open("/proc/cpuinfo") as f:
            phys = core = None
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    ids.add((phys, core))
                    phys = core = None
            if phys is not None and core is not None:
                ids.add((phys, core))
        if ids:
            return len(ids)
    except OSError:
        pass
    return os.cpu_count() or 1


def _expand_slurm_counts(spec: str) -> list[int]:
    """Slurm's compressed per-node task list: '8(x2),7' -> [8, 8, 7]."""
    out = []
    for part in spec.split(","):
        m = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", part.strip())
        if not m:
            return []
        out += [int(m.group(1))] * int(m.group(2) or 1)
    return out


def ranks_on_this_node() -> int | None:
    """Ranks the launcher placed on this node, or None if it did not say.

    The oversubscription check must compare this with the physical cores, not
    the total rank count: SLURM_NTASKS counts every node of the job, and until
    2026-09-22 a correct 16-rank job on two 8-core nodes was refused as
    "16 ranks on 8 cores".
    """
    for var in ("OMPI_COMM_WORLD_LOCAL_SIZE",      # Open MPI
                "MPI_LOCALNRANKS"):                # MPICH, Intel MPI (Hydra)
        v = os.environ.get(var, "")
        if v.isdigit():
            return int(v)
    spec = (os.environ.get("SLURM_STEP_TASKS_PER_NODE")
            or os.environ.get("SLURM_TASKS_PER_NODE"))
    if spec:
        counts = _expand_slurm_counts(spec)
        node = int(os.environ.get("SLURM_NODEID", "0") or 0)
        if 0 <= node < len(counts):
            return counts[node]
    return None


def describe_env(expect_ranks: int | None = None) -> dict:
    rank, size, source = _world()
    slurm = os.environ.get("SLURM_NTASKS")
    if expect_ranks is None and slurm:
        expect_ranks = int(slurm)
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES")
                or os.environ.get("SLURM_NNODES") or 1)
    local = ranks_on_this_node()
    if local is None and nodes == 1:
        local = expect_ranks            # one machine: every rank is local
    return {
        "mpi_source": source,
        "rank": rank,
        "world_size": size,
        "expect_ranks": expect_ranks,
        "nodes": nodes,
        "ranks_on_node": local,
        "threads_in_process": threads_in_this_process(),
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_cores(),
        "slurm_ntasks": slurm,
    }


def require_parallel_env(expect_ranks: int | None = None, *,
                         max_threads: int = 1, verbose: bool = True) -> dict:
    """Check the environment and exit if it is not what the script assumes.

    `expect_ranks` None means "whatever SLURM_NTASKS says", and if that is
    unset the world size is only reported, not checked; a serial script should
    not be forced to declare a rank count.
    """
    env = describe_env(expect_ranks)
    problems = []

    # 1. did the launcher actually build one world?  (2026-08-27)
    if env["expect_ranks"] is not None and env["world_size"] != env["expect_ranks"]:
        problems.append(
            f"world size is {env['world_size']} but {env['expect_ranks']} ranks "
            f"were requested. If it is 1 with many processes running, the "
            f"launcher did not attach MPI and every rank is rank 0 of its own "
            f"world, so every rank guard will pass on every rank.")

    # 2. is each rank single-threaded?  (2026-09-06)
    # A healthy rank still holds RUNTIME_THREADS of the interpreter's and the
    # MPI library's own. The pathology is a thread per core, because the BLAS
    # asked the machine how big it is, so the limit plus that allowance is the
    # whole gate. Until 2026-09-22 it also failed any rank holding as many
    # threads as the node has cores, which refused a healthy 3-thread rank on
    # a 2-core machine.
    n_thr = env["threads_in_process"]
    if n_thr > 0 and max_threads and n_thr > max_threads + RUNTIME_THREADS:
        problems.append(
            f"this rank holds {n_thr} threads against a limit of {max_threads}. "
            f"Set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS to 1 "
            f"and pass them through with mpirun -x, or every rank will spawn a "
            f"thread per core and they will fight. Nothing will error and each "
            f"rank will still report full CPU use.")

    # 2b. are the thread limits set at all?  (2026-09-22)
    # The live count sees only threads that exist when the gate runs. A BLAS
    # built with OpenMP starts them at its first call: under mpirun on WSL with
    # OMP_NUM_THREADS unset, conda-forge numpy held 3 threads at the gate and
    # 18 after one matrix product. (The pip numpy of our GPAW env starts them
    # at import, 18 at the gate, and the count catches that.) The variables
    # are known up front, so check them as well. Neither check covers the
    # other: GPAW sets OMP_NUM_THREADS=1 when imported, so if numpy came first
    # the variable reads 1 while 18 threads already run.
    if max_threads:
        te = env["thread_env"]
        bad = [f"{v}={te[v]}" for v in THREAD_VARS[:3]
               if (te.get(v) or "").strip()
               and not (te[v].strip().isdigit()
                        and 0 < int(te[v]) <= max_threads)]
        if not (te.get("OMP_NUM_THREADS") or "").strip():
            bad.insert(0, "OMP_NUM_THREADS unset")
        if bad:
            problems.append(
                f"thread limit above {max_threads}: {', '.join(bad)}. A BLAS "
                f"built with OpenMP starts its threads at the first "
                f"calculation, so the count above cannot see this yet. Export "
                f"OMP_NUM_THREADS={max_threads} before launching (and pass it "
                f"with mpirun -x on more than one node).")

    # 3. more ranks on this node than it has physical cores?
    # Per node: on a multi-node job whose launcher did not say how many ranks
    # it put here, ranks_on_node is None and this is not checked.
    local = env["ranks_on_node"]
    if (local or 0) > env["physical_cores"]:
        problems.append(
            f"{local} ranks on this node, which has {env['physical_cores']} "
            f"physical cores ({env['logical_cpus']} logical). Open MPI counts a "
            f"slot per physical core and refuses this unless oversubscription "
            f"is forced, and a forced run shares cores between ranks.")

    if verbose and env["rank"] == 0:
        thr = (f"{n_thr} threads/rank" if n_thr > 0 else
               "threads/rank not checked (no /proc on this OS)")
        print(f"[mpi_env_check] {env['mpi_source']}: world {env['world_size']}"
              f", {thr}, {env['physical_cores']} physical cores"
              f" ({env['logical_cpus']} logical)")
    if problems:
        if env["rank"] == 0:
            for p in problems:
                print(f"[mpi_env_check] FAIL: {p}", file=sys.stderr)
        raise SystemExit(2)
    return env


def rank0_write(path, writer, *, barrier: bool = True):
    """Run `writer(path)` on rank 0 only, and raise everywhere if it fails.

    Two accidents in one helper. Writing from every rank corrupts the file
    (2026-09-03); raising on rank 0 alone leaves the others waiting at a
    barrier forever, and a hung job is harder to diagnose than a dead one
    (2026-08-28).
    """
    rank, size, source = _world()
    err = None
    if rank == 0:
        try:
            writer(path)
        except Exception as e:                                   # noqa: BLE001
            err = e
    if barrier and size > 1:
        if source == "gpaw.mpi":
            from gpaw.mpi import world, broadcast
            err = broadcast(err, root=0, comm=world)
            world.barrier()
        elif source == "mpi4py":
            from mpi4py import MPI
            err = MPI.COMM_WORLD.bcast(err, root=0)
            MPI.COMM_WORLD.Barrier()
    if err is not None:
        raise err
    return path


if __name__ == "__main__":
    e = describe_env()
    for k, v in e.items():
        print(f"  {k}: {v}")
    ok = True
    if not (e["thread_env"]["OMP_NUM_THREADS"] or "").strip():
        print("\n  WARNING: OMP_NUM_THREADS is unset. An OpenMP BLAS will start "
              "a thread per core at its first calculation, in every rank.")
        ok = False
    if e["threads_in_process"] < 0:
        print("\n  NOTE: no /proc on this OS, so the thread count is unknown "
              "and require_parallel_env will not check it.")
    elif e["threads_in_process"] > 1 + RUNTIME_THREADS:
        print("\n  WARNING: more than one thread per process. Set "
              "OMP_NUM_THREADS=1 before launching under MPI.")
        ok = False
    if e["logical_cpus"] != e["physical_cores"]:
        print(f"\n  NOTE: nproc reports {e['logical_cpus']} but there are "
              f"{e['physical_cores']} physical cores. Use the latter as the "
              f"rank count.")
    raise SystemExit(0 if ok else 1)
