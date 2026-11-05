#include "ompi_config.h"
//#include "ompi/communicator/communicator.h"
#include <sys/queue.h>
#include <time.h>
#include <sys/time.h>
#include "ompi/mpi/c/bindings.h"


#ifndef QENTRY_H_
#define QENTRY_H_
typedef struct qentry {
    int id;
    char function[30];
    char communicationType[30];
    int blocking;
    char sendDatatype[64];
    char recvDatatype[64];
    int sendcount;
    int64_t sendDatasize;
    int recvcount;
    int64_t recvDatasize;
    char operation[30]; //MPI_Reduce, MPI_Accumulate
    char communicationArea[64];
    MPI_Comm communicator;
    char processorname[30];
    int processrank;
    int partnerrank;
    uint8_t coll_partnerranks[50];
    
    char sendmode[30]; //later
    int immediate; //later
    char usedBtl[30];
    char usedProtocol[30];
    int withinEagerLimit;
    int foundMatchWild;
    char usedAlgorithm[30];
    MPI_Request *request;
    struct timespec start;
    struct timespec end;
    double lateSenderTime;
    double lateReceiverTime;
    double sendWaitingTime;
    double recvWaitingTime;
    //struct timeval start;
    struct timespec initializeRequest;
    struct timespec startRequest;
    //Completion of the first fragment of a long message that requires an acknowledgement
    struct timespec requestCompletePmlLevel;
    //Warten auf Recv-Request
    struct timespec requestWaitCompletion;
    struct timespec requestFini;
    struct timespec sent;//later
    struct timespec bufferFree; //later
    //struct collective_p2p collectives;
    /* For a request that is not finished when the MPI wrapper returns (MPI_Irecv, an MPI_Isend in flight):
     * the wrapper or the PML sets pending_request, the PML sets request_done (and for a receive the number of
     * received bytes) when the request is finished. The entry is not written to the log before that. */
    int pending_request;
    int request_done;
    /* Time the request finished (MPI_Irecv: the message arrived, MPI_Isend: the send request completed) */
    struct timespec done;
    int recv_any_source; /* receive posted with MPI_ANY_SOURCE: partnerrank is the real sender */
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



