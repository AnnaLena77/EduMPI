#include "ompi_config.h"
//#include "ompi/communicator/communicator.h"
#include <sys/queue.h>
#include <time.h>
#include <sys/time.h>
#include "ompi/mpi/c/bindings.h"


#ifndef QENTRY_H_
#define QENTRY_H_
typedef struct qentry {
    /* Hot part: everything an MPI_Isend / MPI_Irecv / MPI_Wait and the completion hooks write, in the first 4 cache lines
     * (the entries are written one by one with other work in between: each cold line costs time and pushes the data of the
     * MPI library out of the caches). The fields below it are only used by collectives and the rest of the library. */
    int id;
    int blocking;
    int sendcount;
    int recvcount;
    int partnerrank;
    int processrank;
    int comm_id;                 /* index of the label of the communicator + 1 (0: none), set by initQentry */
    int withinEagerLimit;
    int foundMatchWild;
    int pending_request;
    int request_done;
    int recv_any_source; /* receive posted with MPI_ANY_SOURCE: partnerrank is the real sender */
    int64_t sendDatasize;
    int64_t recvDatasize;
    MPI_Request *request;
    MPI_Comm communicator;
    struct timespec start;
    struct timespec end;
    /* Time the request finished (MPI_Irecv: the message arrived, MPI_Isend: the send request completed) */
    struct timespec done;
    struct timespec initializeRequest;
    struct timespec requestWaitCompletion;
    struct timespec requestFini;
    char function[30];
    char communicationType[30];
    /* cold part */
    char sendDatatype[64];
    char recvDatatype[64];
    char operation[30]; //MPI_Reduce, MPI_Accumulate
    char communicationArea[64];
    char processorname[30];
    uint8_t coll_partnerranks[50];
    char sendmode[30]; //later
    int immediate; //later
    char usedBtl[30];
    char usedProtocol[30];
    char usedAlgorithm[30];
    double lateSenderTime;
    double lateReceiverTime;
    double sendWaitingTime;
    double recvWaitingTime;
    struct timespec startRequest;
    struct timespec requestCompletePmlLevel;
    struct timespec sent;//later
    struct timespec bufferFree; //later
} qentry;

/* Can the entry be written? (false while it waits for its request) */
static inline int edumpi_request_ready(qentry *e)
{
    return 0 == __atomic_load_n(&e->pending_request, __ATOMIC_RELAXED)
           || 0 != __atomic_load_n(&e->request_done, __ATOMIC_ACQUIRE);
}

/* The PML finished the request: nothing more will be added to the entry */
static inline void edumpi_request_done(qentry *e)
{
    __atomic_store_n(&e->request_done, 1, __ATOMIC_RELEASE);
}

/* ... and it knows how many bytes were received */
static inline void edumpi_request_done_bytes(qentry *e, int64_t recv_bytes)
{
    __atomic_store_n(&e->recvDatasize, recv_bytes, __ATOMIC_RELAXED);
    edumpi_request_done(e);
}
#endif

extern void qentryIntoQueue(qentry **q);
extern void initQentry(qentry **q, int dest, char *function, int function_len, int sendCount, int recvCount, char *commType, int commType_len, MPI_Datatype sendType, MPI_Datatype recvType, MPI_Comm comm, int blocking, MPI_Op op);
extern double timespec_diff(struct timespec start, struct timespec end);
/* Rank of a peer in MPI_COMM_WORLD (the rank inside comm is what the MPI calls carry). */
extern int edumpi_world_rank(MPI_Comm comm, int rank);
//extern void writeIntoFile(qentry q);

extern qentry *q_qentry;

extern qentry* getWritingRingPos(void);

/* Event store and log file (MPI_Init, MPI_Init_thread, MPI_Finalize) */
extern void edumpi_analysis_init_pre(void);
extern void edumpi_analysis_init_post(int provided);
extern void edumpi_analysis_finalize(void);
extern int edumpi_send_completion;



