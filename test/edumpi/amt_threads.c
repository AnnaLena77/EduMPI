/* End-to-end test for EDUMPI_MPI_THREAD_AMT, see README.md in this directory. */
/* 2 ranks, NT threads per rank, each thread does M blocking and M non-blocking exchanges with its own tags */
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef NT
#define NT 4
#endif
#ifndef M
#define M 2000
#endif
static int rank;
static void *worker(void *arg)
{
    int t = (int)(long) arg, peer = 1 - rank, buf = 0;
    for (int i = 0; i < M; i++) {
        if (0 == rank) {
            MPI_Send(&i, 1, MPI_INT, peer, t, MPI_COMM_WORLD);
            MPI_Recv(&buf, 1, MPI_INT, peer, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(&buf, 1, MPI_INT, peer, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&i, 1, MPI_INT, peer, t, MPI_COMM_WORLD);
        }
    }
    for (int i = 0; i < M; i++) {
        MPI_Request r[2];
        MPI_Isend(&i, 1, MPI_INT, peer, 100 + t, MPI_COMM_WORLD, &r[0]);
        MPI_Irecv(&buf, 1, MPI_INT, peer, 100 + t, MPI_COMM_WORLD, &r[1]);
        MPI_Waitall(2, r, MPI_STATUSES_IGNORE);
    }
    return NULL;
}
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided != MPI_THREAD_MULTIPLE) { printf("no MPI_THREAD_MULTIPLE (%d)\n", provided); MPI_Abort(MPI_COMM_WORLD, 1); }
    pthread_t th[NT];
    for (long t = 0; t < NT; t++) pthread_create(&th[t], NULL, worker, (void *) t);
    for (int t = 0; t < NT; t++) pthread_join(th[t], NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == rank) printf("done: %d threads x %d iterations per rank\n", NT, M);
    MPI_Finalize();
    return 0;
}
