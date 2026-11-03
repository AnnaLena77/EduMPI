# EduMPI tests that do not need the GUI

## amt_ring_test.c: the multi-producer ring on its own

No Open MPI build and no database needed:

    cc -O2 -pthread -I../.. amt_ring_test.c -o amt_ring_test && ./amt_ring_test
    cc -O1 -g -fsanitize=thread -pthread -I../.. amt_ring_test.c -o amt_ring_test_tsan && ./amt_ring_test_tsan

It starts 8 producer threads and checks that every event is either emitted
exactly once, in order per producer, or counted as dropped, and it tests the
full ring, the stall timeout and the drain at shutdown.

## amt_threads.c: end to end with a database

Two ranks, four threads per rank, each thread sends and receives with its own
tags (MPI_THREAD_MULTIPLE). Expected per rank: 8000 each of MPI_Send, MPI_Recv,
MPI_Isend and MPI_Irecv, plus one MPI_Barrier. MPI_Waitall is not instrumented.

1. Use a throwaway PostgreSQL (plain PostgreSQL is enough, TimescaleDB is not
   needed) with the first three tables of `EduStore/edustore_schema.sql`
   (`edumpi_runs`, `edumpi_cluster_info`, `edumpi_running_data`).
2. Set `DB_HOST DB_PORT DB_USER DB_NAME DB_PW`, `EDUMPI_PROGRAM` and
   `SLURM_JOBID` (the run id, a number).
3. `mpicc -pthread amt_threads.c -o amt_threads`
4. Run `mpirun --oversubscribe -n 2 ./amt_threads` once with
   `EDUMPI_MPI_THREAD_AMT=1` and once without it, then count the rows per
   function in `edumpi_running_data`.

With the flag the counts are exact. Without it the single-producer ring is
raced by the threads: counts are too low and function names are corrupted.
