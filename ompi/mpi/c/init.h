#include "ompi_config.h"
//#include "ompi/communicator/communicator.h"
#include <sys/queue.h>
#include <time.h>
#include <sys/time.h>
#include "ompi/mpi/c/bindings.h"

extern void enqueue(char** operation, char** datatype, int count, int datasize, char** communicator, int processrank, int partnerrank, time_t ctime);
extern void initializeMongoDB(void);
extern void closeMongoDB(void);
extern pthread_t MONITOR_THREAD;
extern int run_thread;
extern int counter;

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
    /* For a request that is not finished when the MPI wrapper returns (MPI_Irecv):
     * the wrapper sets pending_request, the PML sets request_done (and the number of
     * received bytes) when the request is finished. The monitor thread does not write
     * the entry to the database before that, but not longer than EDUMPI_REQUEST_WAIT_MS. */
    int pending_request;
    int request_done;
    TAILQ_ENTRY(qentry) pointers;
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
//extern void writeIntoFile(qentry q);

extern qentry *q_qentry;

extern qentry* getWritingRingPos(void);

/* Set up / start the event ring and the monitor thread (MPI_Init, MPI_Init_thread) */
extern void edumpi_analysis_init_pre(void);
extern void edumpi_analysis_init_post(int provided);
extern int monitor_started;

#ifndef INIT_H
#define INIT_H
#define MAX_RINGSIZE 1000000
extern qentry *ringbuffer;
extern int writer_pos;
extern int reader_pos;
#endif

extern void closeFile(void);



