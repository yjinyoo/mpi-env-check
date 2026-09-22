"""
Self-test for mpi_env_check.py. Runs without MPI: the world, the thread count
and the core count are replaced, and the launcher variables set by hand.

    python test_mpi_env_check.py

TWO FALSE FAILURES (fixed 2026-09-22, both reproduced under mpirun on WSL first).
  - A correct multi-node job was refused: the total rank count was compared
    with one node's physical cores, so 16 ranks on 2 x 8 cores read as
    "16 ranks on 8 cores".
  - A healthy rank was refused on a 2-core machine: it holds 3 threads of the
    MPI library's own, and any rank with as many threads as cores failed.

ONE GAP (closed 2026-09-22). The live thread count sees only threads that
exist when the gate runs. With OMP_NUM_THREADS unset under mpirun on WSL,
conda-forge numpy held 3 threads at the gate and 18 after its first matrix
product, because an OpenMP BLAS starts its pool at the first call. (The pip
numpy of our GPAW env starts it at import, so there the count did catch the
2026-09-06 failure.) The thread-limit variables are now checked as well.

THE FAILURES THE GATE EXISTS FOR must still fail.
  - 2026-08-27: 32 ranks, each rank 0 of a world of size 1.
  - 2026-09-06: 33 BLAS threads per rank on 16 cores, and the unset variable
    that causes it, seen before the threads exist.
  - More ranks on one node than it has physical cores.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpi_env_check as m                                        # noqa: E402

LAUNCHER_VARS = ("SLURM_NTASKS", "SLURM_JOB_NUM_NODES", "SLURM_NNODES",
                 "SLURM_STEP_TASKS_PER_NODE", "SLURM_TASKS_PER_NODE",
                 "SLURM_NODEID", "OMPI_COMM_WORLD_LOCAL_SIZE",
                 "MPI_LOCALNRANKS") + m.THREAD_VARS


@contextlib.contextmanager
def machine(*, world_size, threads, cores, env=None):
    """This process as rank 0 of `world_size`, on a node of `cores` cores.

    OMP_NUM_THREADS=1 unless `env` says otherwise; None in `env` means unset.
    """
    saved_env = {k: os.environ.pop(k, None) for k in LAUNCHER_VARS}
    saved = (m._world, m.threads_in_this_process, m.physical_cores)
    for k, v in {"OMP_NUM_THREADS": "1", **(env or {})}.items():
        if v is not None:
            os.environ[k] = v
    m._world = lambda: (0, world_size, "mock")
    m.threads_in_this_process = lambda: threads
    m.physical_cores = lambda: cores
    try:
        yield
    finally:
        m._world, m.threads_in_this_process, m.physical_cores = saved
        for k in LAUNCHER_VARS:
            os.environ.pop(k, None)
            if saved_env[k] is not None:
                os.environ[k] = saved_env[k]


def gate(**kw) -> tuple[bool, str]:
    """(passed, what it printed)."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            m.require_parallel_env(**kw)
        return True, out.getvalue()
    except SystemExit:
        return False, out.getvalue()


# ---- the two false failures ---------------------------------------------------

def test_multinode_job_passes_with_open_mpi():
    with machine(world_size=16, threads=3, cores=8,
                 env={"SLURM_NTASKS": "16", "SLURM_JOB_NUM_NODES": "2",
                      "OMPI_COMM_WORLD_LOCAL_SIZE": "8"}):
        ok, msg = gate(max_threads=1)
    assert ok, msg


def test_multinode_job_passes_with_srun():
    with machine(world_size=16, threads=3, cores=8,
                 env={"SLURM_NTASKS": "16", "SLURM_NNODES": "2",
                      "SLURM_STEP_TASKS_PER_NODE": "8(x2)",
                      "SLURM_NODEID": "1"}):
        ok, msg = gate(max_threads=1)
    assert ok, msg


def test_multinode_job_passes_when_launcher_is_silent():
    # No per-node count at all: the per-node check is skipped, not guessed.
    with machine(world_size=16, threads=3, cores=8,
                 env={"SLURM_NTASKS": "16", "SLURM_JOB_NUM_NODES": "2"}):
        ok, msg = gate(max_threads=1)
    assert ok, msg


def test_healthy_rank_on_two_cores_passes():
    with machine(world_size=2, threads=3, cores=2):
        ok, msg = gate(expect_ranks=2, max_threads=1)
    assert ok, msg


# ---- what the gate is for ------------------------------------------------------

def test_singleton_worlds_fail():                                # 2026-08-27
    with machine(world_size=1, threads=3, cores=64,
                 env={"SLURM_NTASKS": "32"}):
        ok, msg = gate(max_threads=1)
    assert not ok and "world size is 1" in msg, msg


def test_thread_per_core_fails():                                # 2026-09-06
    with machine(world_size=8, threads=33, cores=16):
        ok, msg = gate(expect_ranks=8, max_threads=1)
    assert not ok and "33 threads" in msg, msg


def test_unset_thread_limit_fails_before_blas_starts():         # 2026-09-06
    # 3 threads is what a rank holds before an OpenMP BLAS has run once;
    # the unset variable is the only thing visible this early.
    with machine(world_size=8, threads=3, cores=16,
                 env={"OMP_NUM_THREADS": None}):
        ok, msg = gate(expect_ranks=8, max_threads=1)
    assert not ok and "OMP_NUM_THREADS unset" in msg, msg


def test_threads_already_running_fail_though_variable_reads_one():
    # GPAW imported after numpy: GPAW has set OMP_NUM_THREADS=1, but the pip
    # OpenBLAS started 18 threads at numpy's import. Only the count sees it.
    with machine(world_size=8, threads=18, cores=8,
                 env={"OMP_NUM_THREADS": "1"}):
        ok, msg = gate(expect_ranks=8, max_threads=1)
    assert not ok and "18 threads" in msg, msg


def test_thread_limit_above_max_fails():
    with machine(world_size=8, threads=3, cores=16,
                 env={"OPENBLAS_NUM_THREADS": "8"}):
        ok, msg = gate(expect_ranks=8, max_threads=1)
    assert not ok and "OPENBLAS_NUM_THREADS=8" in msg, msg


def test_hybrid_limit_within_max_passes():
    with machine(world_size=4, threads=7, cores=16,
                 env={"OMP_NUM_THREADS": "4"}):
        ok, msg = gate(expect_ranks=4, max_threads=4)
    assert ok, msg


def test_max_threads_zero_turns_thread_checks_off():
    with machine(world_size=4, threads=40, cores=16,
                 env={"OMP_NUM_THREADS": None}):
        ok, msg = gate(expect_ranks=4, max_threads=0)
    assert ok, msg


def test_oversubscribed_node_fails():
    with machine(world_size=16, threads=3, cores=8,
                 env={"OMPI_COMM_WORLD_LOCAL_SIZE": "16"}):
        ok, msg = gate(expect_ranks=16, max_threads=1)
    assert not ok and "16 ranks on this node" in msg, msg


def test_oversubscribed_single_machine_fails_without_launcher_vars():
    with machine(world_size=16, threads=3, cores=8):
        ok, msg = gate(expect_ranks=16, max_threads=1)
    assert not ok and "16 ranks on this node" in msg, msg


def test_uneven_srun_layout_reads_this_node():
    with machine(world_size=16, threads=3, cores=8,
                 env={"SLURM_NNODES": "2", "SLURM_STEP_TASKS_PER_NODE": "9,7",
                      "SLURM_NODEID": "0"}):
        ok, msg = gate(expect_ranks=16, max_threads=1)
    assert not ok and "9 ranks on this node" in msg, msg


# ---- pieces --------------------------------------------------------------------

def test_slurm_count_expansion():
    assert m._expand_slurm_counts("8(x2),7") == [8, 8, 7]
    assert m._expand_slurm_counts("16") == [16]
    assert m._expand_slurm_counts("8(x2),junk") == []


def test_unknown_thread_count_is_said_not_passed_silently():
    with machine(world_size=4, threads=-1, cores=8):
        ok, msg = gate(expect_ranks=4, max_threads=1)
    assert ok and "not checked" in msg, msg


def test_rank0_write_writes_and_raises():
    saved = m._world
    m._world = lambda: (0, 1, "serial")
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.txt")
            m.rank0_write(p, lambda q: open(q, "w").write("x"))
            assert open(p).read() == "x"

            def broken(q):
                raise OSError("disk full")
            try:
                m.rank0_write(p, broken)
            except OSError as e:
                assert "disk full" in str(e)
            else:
                raise AssertionError("rank0_write swallowed the writer's error")
    finally:
        m._world = saved


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}\n        {type(e).__name__}: "
                  f"{str(e).strip()[:300]}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
