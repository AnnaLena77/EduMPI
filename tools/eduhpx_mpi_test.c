/* Test of the measuring MPI: non-blocking point-to-point (eager and rendezvous), MPI_ANY_SOURCE, a receive that never
 * completes, collectives, and a microbenchmark of the call overhead. Each rank prints how many Isend/Irecv it called.
 * usage: eduhpx_mpi_test [iterations] */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int iters = argc > 1 ? atoi(argv[1]) : 2000;
    long n_isend = 0, n_irecv = 0;
    int sizes[3] = {8, 65536, 1 << 20};
    char *sbuf = malloc(1 << 20), *rbuf = malloc(1 << 20);
    memset(sbuf, rank, 1 << 20);
    int right = (rank + 1) % size, left = (rank + size - 1) % size;

    /* 1. ring exchange, three message sizes, specific source */
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < 20; i++) {
            MPI_Request rq[2];
            MPI_Irecv(rbuf, sizes[s], MPI_CHAR, left, 10 + s, MPI_COMM_WORLD, &rq[0]); n_irecv++;
            MPI_Isend(sbuf, sizes[s], MPI_CHAR, right, 10 + s, MPI_COMM_WORLD, &rq[1]); n_isend++;
            MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
        }
    }
    /* 2. receives with MPI_ANY_SOURCE: everybody sends to rank 0 */
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Request r;
            MPI_Irecv(rbuf, 4096, MPI_CHAR, MPI_ANY_SOURCE, 77, MPI_COMM_WORLD, &r); n_irecv++;
            MPI_Wait(&r, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Request r;
        MPI_Isend(sbuf, 4096, MPI_CHAR, 0, 77, MPI_COMM_WORLD, &r); n_isend++;
        MPI_Wait(&r, MPI_STATUS_IGNORE);
    }
    /* 3. a receive that is posted and never gets a message (cancelled at the end) */
    MPI_Request idle;
    MPI_Irecv(rbuf, 16, MPI_CHAR, MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, &idle); n_irecv++;
    /* 4. collectives */
    MPI_Bcast(sbuf, 65536, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    /* 5. microbenchmark: ring of 8-byte messages, time per Isend call and per Isend+Irecv+Waitall */
    double t_isend = 0;
    double t0 = now();
    for (int i = 0; i < iters; i++) {
        MPI_Request rq[2];
        MPI_Irecv(rbuf, 8, MPI_CHAR, left, 5, MPI_COMM_WORLD, &rq[0]); n_irecv++;
        double a = now();
        MPI_Isend(sbuf, 8, MPI_CHAR, right, 5, MPI_COMM_WORLD, &rq[1]); n_isend++;
        t_isend += now() - a;
        MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
    }
    double t1 = now();
    MPI_Barrier(MPI_COMM_WORLD);
    printf("RANK %d isend_calls=%ld irecv_calls=%ld  bench: %.3f us per ring step, %.3f us per Isend call\n",
           rank, n_isend, n_irecv, (t1 - t0) / iters * 1e6, t_isend / iters * 1e6);
    if (argc > 2 && 0 == strcmp(argv[2], "idle")) {
        /* leave the receive posted: the record is flagged NOT_COMPLETED */
    } else {
        MPI_Cancel(&idle);
        MPI_Request_free(&idle);
    }
    MPI_Finalize();
    return 0;
}
