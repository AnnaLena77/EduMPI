/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2018 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2007-2008 Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2025      Advanced Micro Devices, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <semaphore.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <sys/queue.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/mman.h>

#ifdef ENABLE_ANALYSIS
#include <libpq-fe.h>
#include <pthread.h>
#include <sched.h>
#include <hwloc.h>
#endif

#include "opal/util/show_help.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/mpi/c/bindings.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/constants.h"
#include "ompi/mpi/c/pgcopy.h"

#ifdef ENABLE_ANALYSIS
#include "ompi/mpi/c/init.h"
#include "ompi/mpi/c/edumpi_ring.h"
#include "ompi/peruse/peruse.h"
#endif


#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Init = PMPI_Init
#endif
#define MPI_Init PMPI_Init
#endif

static struct timeval start;
static struct timeval init_sql_finished;
static struct timeval init_sql_start;

static int ID=0;
static char *job_id;
static struct timespec start_runtime;
static int count_q_entry = 0;
static int queue_lock;
static TAILQ_HEAD(tailhead, qentry) head;
static int queue_length=0;
static int lock = 0;
//sem_t ENSURE_INIT;

static int main_core, logging_core;

static int size, processrank;
static char processrank_arr[4];
char proc_name[MPI_MAX_PROCESSOR_NAME];
int proc_name_length;

static FILE *file;
static char filename[20];
static char filename[20];
static char* mapped_data;
static int file_size;
static int illi=1;

static int count_before = 0;
static int partner_before = 0;
static char count_before_arr[10];
static char partner_before_arr[10];

qentry *q_qentry;
qentry *ringbuffer;
int writer_pos;
int reader_pos;

/* AMT mode (EDUMPI_MPI_THREAD_AMT=1): the program calls MPI from many threads at
 * the same time, so events go into a multi-producer ring (edumpi_ring.h) instead
 * of the single-producer ring above. */
static int amt_mode = 0;
static edumpi_amt_ring_t amt_ring;
int monitor_started = 0;

/* How long (seconds) an entry waits for its request to finish (MPI_Irecv, see
 * pending_request in init.h) before it is written without the received size. */
static double request_wait_s = 2.0;

/* ticket of the entries of the single-producer ring (the multi-producer ring numbers its slots
 * itself): see lifecycle in init.h */
static uint64_t edumpi_ticket_counter = 0;

/* EDUMPI_RECV_DURATION=1: the time_diff of an MPI_Irecv is the time from posting the receive until it completed
 * (the time the receive request was pending). Without it time_diff of a receive is 0, as before. */
static int edumpi_recv_duration = 0;

/* EDUMPI_SEND_COMPLETION=1 (AMT mode only): for an MPI_Isend that is still in flight when the call returns, an
 * extra entry "MPI_Isend_done" is written when the send request completes (see pml_ucx_request.h). Read by the
 * UCX PML, therefore not static. */
int edumpi_send_completion = 0;

double timespec_diff(struct timespec start, struct timespec end) {
    long long elapsed_sec = end.tv_sec - start.tv_sec;
    long long elapsed_ns = end.tv_nsec - start.tv_nsec;

    // Korrigiere Nanosekunden, falls Überlauf auftritt
    if (elapsed_ns < 0) {
        elapsed_sec -= 1;  // Eine Sekunde abziehen
        elapsed_ns += 1000000000;  // 1 Sekunde in Nanosekunden hinzufügen
    }
    // Gesamtzeit in Sekunden berechnen
    return (double)elapsed_sec + (double)elapsed_ns / 1e9;
}

const char* get_mpi_op_name(MPI_Op op) {
    if (op == MPI_SUM) {
        return "MPI_SUM";
    } else if (op == MPI_MAX) {
        return "MPI_MAX";
    } else if (op == MPI_MIN) {
        return "MPI_MIN";
    } else if (op == MPI_PROD) {
        return "MPI_PROD";
    } else if (op == MPI_LAND) {
        return "MPI_LAND";
    } else if (op == MPI_BAND) {
        return "MPI_BAND";
    } else if (op == MPI_LOR) {
        return "MPI_LOR";
    } else if (op == MPI_BOR) {
        return "MPI_BOR";
    } else if (op == MPI_LXOR) {
        return "MPI_LXOR";
    } else if (op == MPI_BXOR) {
        return "MPI_BXOR";
    } else {
        return "UNKNOWN_OP";
    }
}

void convert_MPI_datatype(MPI_Datatype type, char *datatype_field){
    if(type == MPI_INT) {
        memcpy(datatype_field, "MPI_INT", 7);
    } else if(type == MPI_CHAR){
        memcpy(datatype_field, "MPI_CHAR", 8);
    } else if (type == MPI_DOUBLE){
        memcpy(datatype_field, "MPI_DOUBLE", 10);
    } else if (type == MPI_LONG){
        memcpy(datatype_field, "MPI_LONG", 8);
    } else {
        //int type_name_length;
        //MPI_Type_get_name(type, datatype_field, &type_name_length);
        //datatype_field[type_name_length] = '\0'; 
    }
}

/* Label of a communicator for the database: MPI_COMM_WORLD, a name set by the program
 * (MPI_Comm_set_name), or the name Open MPI gives when it creates the communicator
 * ("MPI COMM 3 DUP FROM 0"). MPI_Comm_get_name reports the latter only after a name was set,
 * so those communicators used to be stored with an empty label. */
static void edumpi_comm_label(MPI_Comm comm, char *out, size_t outlen){
    char base[64];
    out[0] = '\0';
    if(comm == NULL || comm == MPI_COMM_NULL){
        return;
    }
    if(comm == MPI_COMM_WORLD){
        snprintf(out, outlen, "MPI_COMM_WORLD");
        return;
    }
    if(comm->c_name != NULL && comm->c_name[0] != '\0'){
        snprintf(base, sizeof(base), "%s", comm->c_name);
    } else {
        snprintf(base, sizeof(base), "MPI COMM %d", (int) comm->c_index);
    }
    /* The disjoint parts of a split (even and odd ranks, say) have the same context id and so the
     * same name. Add the world rank of the first member to tell them apart. */
    if(!OMPI_COMM_IS_INTER(comm) && ompi_comm_size(comm) < ompi_comm_size(MPI_COMM_WORLD)){
        snprintf(out, outlen, "%.52s @%d", base, edumpi_world_rank(comm, 0));
    } else {
        snprintf(out, outlen, "%s", base);
    }
}

int edumpi_world_rank(MPI_Comm comm, int rank){
    if(rank < 0 || comm == NULL || comm == MPI_COMM_NULL || comm == MPI_COMM_WORLD){
        return rank;
    }
    /* ompi_comm_remote_size() is 0 for an intra-communicator */
    int peers = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
    if(rank >= peers){
        return rank;
    }
    ompi_proc_t *proc = ompi_comm_peer_lookup(comm, rank);
    if(proc == NULL){
        return rank;
    }
    return (int) proc->super.proc_name.vpid;
}

void initQentry(qentry **q, int dest, char *function, int function_len, int sendCount, int recvCount, char *commType, int commType_len, MPI_Datatype sendType, MPI_Datatype recvType, MPI_Comm comm, int blocking, MPI_Op op){
    if(q==NULL || *q==NULL){
    	return;
    } else {
        qentry *item = *q;
        item->id = 0;
        memcpy(item->function, function, function_len);
        item->blocking = blocking;
        
        if(sendType == NULL){
            memcpy(item->sendDatatype, "", 0);
        } else {
            convert_MPI_datatype(sendType, item->sendDatatype);
        }
        if(recvType == NULL){
            memcpy(item->recvDatatype, "", 0);
        } else {
            convert_MPI_datatype(recvType, item->recvDatatype);
        }
        
        memcpy(item->communicationType, commType, commType_len);
        
        item->sendcount = sendCount;
        item->sendDatasize = sendCount * sizeof(sendType);
        item->recvcount = recvCount;
        item->recvDatasize = recvCount * sizeof(recvType);
        
        if(op == NULL){
            memcpy(item->operation, "", 0);
        } else {
            memcpy(item->operation, get_mpi_op_name(op), 10);
        }
        
        edumpi_comm_label(comm, item->communicationArea, sizeof(item->communicationArea));
        if(comm != NULL){
            item->communicator = comm;
        }
        memcpy(item->processorname, proc_name, proc_name_length);
        item->processrank = processrank;
        item->partnerrank = edumpi_world_rank(comm, dest);
        
        //printf("Test Rank: %d, WriterPos: %d, Bcast: %p\n", processrank, writer_pos, item);
        
        item->recvWaitingTime = 0.0;
        item->sendWaitingTime = 0.0;

        item->lateSenderTime = 0.0;
        item->lateReceiverTime = 0.0;
        
        item->request = NULL;
        item->end.tv_sec = 0;
        item->end.tv_nsec = 0;
        item->initializeRequest.tv_sec = 0;      /* post time of a receive, set by the UCX completion hook */
        item->initializeRequest.tv_nsec = 0;
       
        memcpy(item->sendmode, "", 0);
        item->immediate = 0;
        memcpy(item->usedBtl, "", 0);
        memcpy(item->usedProtocol, "", 0);
        item->withinEagerLimit = -1;
        item->foundMatchWild = -1;
        memcpy(item->usedAlgorithm, "", 0);
        memset(item->coll_partnerranks, 0, 50);

    }
}


void qentryIntoQueue(qentry **q){
    //printf("IntoQueue: %d\n", count_q_entry++);
    //printf("Echtes Samplerandom: %d\n",samplerandom);
    //if(time(NULL)-counter_time>0.1){ }
    qentry *item = *q;
    //printf("%s\n", item->function);
    //gettimeofday(&item->intoQueue, NULL);
    item->id = ++ID;
    //printf("intoqueue: %d \n", item->id);
    TAILQ_INSERT_TAIL(&head, item, pointers);
    //printf("Es kam was rein\n");
    queue_length++;
    //printf("Queue Length: %d\n", queue_length);
    //printf("Leer? %d\n", TAILQ_EMPTY(&head));
}

qentry* dequeue(){
    qentry *item;
    item = TAILQ_FIRST(&head);
    if(item != NULL){
        TAILQ_REMOVE(&head, item, pointers);
    }
    return item;
}

//Needs to be global!
pthread_t MONITOR_THREAD = 0;
int run_thread;
int counter;

static void registerCluster(){
    
    MPI_Comm comm = MPI_COMM_WORLD;
    int size, processrank;
    char processorname[30];
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &processrank);
    
    int proc_name_length;
    MPI_Get_processor_name(processorname, &proc_name_length);
}

/*static void writeToPostgres(PGconn *conn, char* buffer, int offset){
    counter ++;
    const char copyQuery[256];
    snprintf("COPY " + db_name + " FROM STDIN (FORMAT binary)";
    PGresult *res;
    res = PQexec(conn, copyQuery);
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        printf("Fehler beim Starten des COPY-Befehls: %s\n", PQresultErrorMessage(res));
        PQclear(res);
        PQfinish(conn);
        return;
    }
    PQclear(res);
    PQputCopyData(conn, PGCOPY_HEADER, 19);
    
    int copyRes = PQputCopyData(conn, buffer, offset);
    // Beende den COPY-Befehl
    if(copyRes == 1){
        if(PQputCopyEnd(conn, NULL)==1){
            res = PQgetResult(conn);
            if(PQresultStatus(res) == PGRES_COMMAND_OK){
                printf("Insert a record successfully\n");
            }
        }
    }
    PQflush(conn);
    
}*/

void print_timespec(struct timespec ts) {
    // Zeit in Sekunden in eine tm-Struktur umwandeln
    struct tm *timeinfo;
    timeinfo = localtime(&ts.tv_sec);
    
    // Stunden, Minuten und Sekunden aus der tm-Struktur
    int hours = timeinfo->tm_hour;
    int minutes = timeinfo->tm_min;
    int seconds = timeinfo->tm_sec;
    
    // Ausgabe im Format yyyy-mm-dd hh:mm:ss.nnnnnnnnn
    printf(" %04d-%02d-%02d %02d:%02d:%02d.%09ld\n",
           timeinfo->tm_year + 1900,
           timeinfo->tm_mon + 1,
           timeinfo->tm_mday,
           hours, minutes, seconds, ts.tv_nsec);
}

void qentryToBinary(qentry *q, char *buffer, int *off){
        counter ++;
        //printf("Counter: %d\n", counter);
        qentry *item = q;
        int offset = *off;
            
        newRow(buffer, 16, &offset);

        //printf("%.9f Seconds\n", item->sendWaitingTime);
        
        
        int job = atoi(job_id);
        intToBinary(job, buffer, &offset);

        stringToBinary(item->function, buffer, &offset);
        
        stringToBinary(item->communicationType, buffer, &offset);
        
        int64ToBinary(item->sendDatasize, buffer, &offset);
        
        int64ToBinary(item->recvDatasize, buffer, &offset);
        
        stringToBinary(item->communicationArea, buffer, &offset);
        
        stringToBinary(item->processorname, buffer, &offset);
        
        intToBinary(item->processrank, buffer, &offset);
        
        intToBinary(item->partnerrank, buffer, &offset);
        
        stringToBinary(item->usedAlgorithm, buffer, &offset);

        timestampToBinary(item->start, buffer, &offset, 1);
        
        timestampToBinary(item->end, buffer, &offset, 0);
        
        doubleToBinary(item->lateSenderTime, buffer, &offset);
        
        doubleToBinary(item->lateReceiverTime, buffer, &offset);
        
        double time_diff = timespec_diff(item->start, item->end);
        if (edumpi_recv_duration && 0 == strcmp(item->function, "MPI_Irecv")
            && (item->initializeRequest.tv_sec != 0 || item->initializeRequest.tv_nsec != 0)) {
            double pending = timespec_diff(item->initializeRequest, item->end);
            if (pending > 0) {
                time_diff = pending;
            }
        }
        doubleToBinary(time_diff, buffer, &offset);
        
        byteaToBinary(item->coll_partnerranks, 400, buffer, &offset, strcmp(item->communicationType, "collective"));
        
        *off = offset;
        item = NULL;
        
}

#define TIME_TO_WAIT 0.5

/* Wall-clock time in seconds. clock() is the CPU time of all threads of the
 * process, which runs faster than real time in a multi-threaded program. */
static double edumpi_wall_seconds(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Age of an entry in seconds: the time since its call returned (end is a CLOCK_REALTIME stamp). */
static double edumpi_entry_age_seconds(qentry *e){
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return (double)(now.tv_sec - e->end.tv_sec) + (double)(now.tv_nsec - e->end.tv_nsec) * 1e-9;
}

/* Short sleep for the monitor thread while the ring is empty. (sleep(0.1)
 * takes an unsigned int, so it was sleep(0): a busy loop that kept one core
 * at 100%.) */
static void edumpi_idle_sleep(void){
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
}

/* AMT mode: hand one entry of the multi-producer ring to the COPY stream */
#define AMT_DRAIN_BATCH 4096
static void amt_emit(qentry *entry, void *ctx){
    char buffer[2048];
    int offset = 0;
    qentryToBinary(entry, buffer, &offset);
    PQputCopyData((PGconn *) ctx, buffer, offset);
}

//Monitor-Function for SQL-Connection
static void* SQLMonitorFunc(void* _arg){
    
    char *db_host = getenv("DB_HOST");
    char *db_port = getenv("DB_PORT");
    char *db_user = getenv("DB_USER");
    char *db_name = getenv("DB_NAME");
    char *db_pw = getenv("DB_PW");
    
    if (db_host == NULL || db_port == NULL || db_name == NULL || db_user == NULL || db_pw == NULL) {
        fprintf(stderr, "One or more environment variables are not set\n");
        return NULL;
    }
    
    int conninfo_size=(36 + strlen(db_host) + strlen(db_port) + strlen(db_name) + strlen(db_user) + strlen(db_pw));
    char *conninfo = malloc(conninfo_size);
    
    strcpy(conninfo, "host=");
    strcat(conninfo, db_host);
    strcat(conninfo, " port=");
    strcat(conninfo, db_port);
    strcat(conninfo, " dbname=");
    strcat(conninfo, db_name);
    strcat(conninfo, " user=");
    strcat(conninfo, db_user);
    strcat(conninfo, " password=");
    strcat(conninfo, db_pw);
    

    PGconn *conn = PQconnectdb(conninfo);
    bool check_job_id=0;
       
    if(processrank==0){
        char *user_id = strrchr(getenv("HOME"), '/') + 1;
        char start_time_str[30]; // Platz für den Zeitstempel
        strftime(start_time_str, sizeof(start_time_str), "%Y-%m-%d %H:%M:%S", localtime(&(start_runtime.tv_sec)));
        char *program_name = getenv("EDUMPI_PROGRAM");
        const char *query = "INSERT INTO edumpi_runs (edumpi_run_id, start_time, user_id, program_name) VALUES ($1, $2, $3, $4)";
        const char *paramValues[4] = {job_id, start_time_str, user_id, program_name};
        int paramlengths[4] = {strlen(job_id), strlen(start_time_str), strlen(user_id), strlen(program_name)};
        PGresult *res = PQexecParams(conn, query, 4, NULL, paramValues, paramlengths, NULL, 0);
        PQclear(res);  
    }
    else {     
       while(!check_job_id){
           const char *query = "SELECT edumpi_run_id FROM edumpi_runs WHERE edumpi_run_id = $1";
	  const char *paramValues[1] = {job_id};
           PGresult *res = PQexecParams(conn, query, 1, NULL, paramValues, (const int*)strlen(job_id), NULL, 0);
           check_job_id = (PQntuples(res));
           PQclear(res);
  
       }
   }

    
       char proc_rank[4];
       sprintf(proc_rank, "%d", processrank);
       const char *query = "INSERT INTO edumpi_cluster_info (edumpi_run_id, processorname, processrank) VALUES ($1, $2, $3)";
       const char *paramValues[3] = {job_id ,proc_name, proc_rank};
       PGresult *res_ = PQexecParams(conn, query, 3, NULL, paramValues, (const int*)strlen(job_id)+strlen(proc_name)+strlen(proc_rank), NULL, 0);
       PQclear(res_);
       
    
    double start = edumpi_wall_seconds();
    int held = 0;            /* the next entry waits for its request ...     */
    double held_since = 0.0; /* ... since this time (default mode only)      */
    
    const char *copyQuery = "COPY edumpi_running_data FROM STDIN (FORMAT binary)";
    PGresult *res;
    res = PQexec(conn, copyQuery);
    PQclear(res);
    PQputCopyData(conn, PGCOPY_HEADER, 19);
    
   // int copyRes = PQputCopyData(conn, buffer, offset);
    //int offset = 0;
    //char *buffer = (char*)malloc(sizeof(char)*20000000);

    while(run_thread){
        /* Default mode: hold the next entry back while its request is not finished */
        int hold = 0;
        if(!amt_mode && run_thread && writer_pos != -1 && reader_pos != writer_pos
           && !(reader_pos == MAX_RINGSIZE-1 && writer_pos == 0)){
            int next = (reader_pos == MAX_RINGSIZE-1) ? 0 : reader_pos + 1;
            qentry *cand = &ringbuffer[next];
            if(edumpi_request_ready(cand)){
                held = 0;
            } else if(cand->end.tv_sec != 0 || cand->end.tv_nsec != 0){
                /* The call returned, the request has not finished. The wait is measured from
                 * the time the receive was posted, so a pool of idle receives is not waited
                 * for one after the other. */
                held = 0;
                hold = (edumpi_entry_age_seconds(cand) < request_wait_s)
                       || ((__atomic_load_n(&cand->lifecycle, __ATOMIC_ACQUIRE) & 3) == EDUMPI_LC_FINISHING);
            } else {
                /* the call itself has not returned */
                double now_held = edumpi_wall_seconds();
                if(!held){
                    held = 1;
                    held_since = now_held;
                }
                hold = (now_held - held_since < request_wait_s);
            }
        }

        if(amt_mode){
            if(0 == edumpi_amt_ring_drain(&amt_ring, amt_emit, conn, 0, AMT_DRAIN_BATCH)){
                edumpi_idle_sleep();
            }
        }
        else if(hold){
            edumpi_idle_sleep();
        }
        else if(writer_pos == -1 || reader_pos==writer_pos || (reader_pos == MAX_RINGSIZE-1 && writer_pos == 0)){
            //printf("Reader sleeps at Position %d, %d\n", reader_pos, writer_pos);
            edumpi_idle_sleep();
            //printf("Prozess: %d haengt beim Lesen\n", processrank);
            if(!run_thread) break;
        }
        else {
            if(reader_pos == MAX_RINGSIZE-1){
                reader_pos = 0;
            }
            else{
                reader_pos ++;
            }
            
            char buffer[2048];
            int offset = 0;
            qentry *item = &ringbuffer[reader_pos];
            if(!edumpi_request_ready(item) && run_thread
               && (item->end.tv_sec != 0 || item->end.tv_nsec != 0)
               && edumpi_lifecycle_try_abandon(item)){
                /* A UCX receive that waited too long: do not write it without data. It is
                 * written as a new entry when its message arrives. */
                held = 0;
                goto entry_done;
            }
            if(reader_pos == writer_pos){
                while(item->end.tv_nsec == 0){
                    edumpi_idle_sleep();
                    //printf("Prozess: %d haengt beim Lesen\n", processrank);
                    if(!run_thread) break;
	   }
            }
            //printf("Datenbankvorbereitung fuer: %s, Rank: %d\n", item->function, processrank);
            /*MPI_Status status;
            MPI_Request *request = item->request;
            int flag = 0;
            if(request == NULL){
                printf("1: request is NULL\n");
            }
            if(request == MPI_REQUEST_NULL){
                printf("1: request is MPI_REQUEST_NULL\n");
            }
            
            if(request != MPI_REQUEST_NULL && request != NULL && strcmp(item->function, "MPI_Wait")){
                while (!flag) {
                    printf("Test1\n");
	       MPI_Test(request, &flag, &status);
	       usleep(100);
	   }
            }*/
            qentryToBinary(&ringbuffer[reader_pos], buffer, &offset);
            PQputCopyData(conn, buffer, offset);
            held = 0;
            entry_done: ;
        }
            
        double current = edumpi_wall_seconds();
        if(current >= start + TIME_TO_WAIT){
            PQputCopyEnd(conn, NULL);
            PQflush(conn);
            while ((res = PQgetResult(conn)) != NULL) {
                if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                    fprintf(stderr, "COPY into edumpi_running_data failed: %s\n", PQresultErrorMessage(res));
                }
                PQclear(res);
            }
            res = PQexec(conn, copyQuery);
            
            if (PQresultStatus(res) != PGRES_COPY_IN) {
                fprintf(stderr, "COPY start failed: %s\n", PQresultErrorMessage(res));
                //return -1;
	}
            
            PQclear(res);
            PQputCopyData(conn, PGCOPY_HEADER, 19);
            start = current;
        }
    }
  
    if(amt_mode){
        /* shutdown: emit everything that is left, complete or not */
        while(edumpi_amt_ring_drain(&amt_ring, amt_emit, conn, 1, AMT_DRAIN_BATCH) > 0){
        }
        uint64_t dropped = edumpi_amt_ring_dropped(&amt_ring);
        if(dropped > 0){
            fprintf(stderr, "EduMPI (rank %d): %llu events were dropped because the ring buffer was full\n",
                    processrank, (unsigned long long) dropped);
        }
    }
    while(!amt_mode && reader_pos != writer_pos){
        if(reader_pos == MAX_RINGSIZE-1){
            reader_pos = 0;
        } else {
            reader_pos ++;
        }
        char buffer[2048];
        int offset = 0;
        
        qentry *item = &ringbuffer[reader_pos];
        
        /*MPI_Status status;
        MPI_Request *request = item->request;
        int flag = 0;
        if(request == NULL){
                printf("2: request is NULL\n");
            }
            if(request == MPI_REQUEST_NULL){
                printf("2: request is MPI_REQUEST_NULL\n");
            }
            
        
        if(request != MPI_REQUEST_NULL && request != NULL && strcmp(item->function, "MPI_Wait")){
            while (!flag) {
                printf("Test2\n");
	   MPI_Test(request, &flag, &status);
	   usleep(100);
            }
        }*/
        
        qentryToBinary(&ringbuffer[reader_pos], buffer, &offset);
        PQputCopyData(conn, buffer, offset);
        double current = edumpi_wall_seconds();
        if(current >= start + TIME_TO_WAIT){
            PQputCopyEnd(conn, NULL);
            PQflush(conn);
            while ((res = PQgetResult(conn)) != NULL) {
                if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                    fprintf(stderr, "COPY into edumpi_running_data failed: %s\n", PQresultErrorMessage(res));
                }
                PQclear(res);
            }
            res = PQexec(conn, copyQuery);
            PQclear(res);
            PQputCopyData(conn, PGCOPY_HEADER, 19);
            
            start = current;
        }
    }
    PQputCopyEnd(conn, NULL);
    PQflush(conn);
    while ((res = PQgetResult(conn)) != NULL) {
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            fprintf(stderr, "COPY into edumpi_running_data failed: %s\n", PQresultErrorMessage(res));
        }
        PQclear(res);
    }
    
    if(processrank == 0){
        struct timespec end_runtime;
        clock_gettime(CLOCK_REALTIME, &end_runtime);

        // Zeitstempel für end_time als String erstellen
        char end_time_str[30];
        strftime(end_time_str, sizeof(end_time_str), "%Y-%m-%d %H:%M:%S", localtime(&(end_runtime.tv_sec)));

        // SQL-Abfrage für UPDATE der end_time
        const char *update_query = "UPDATE edumpi_runs SET end_time = $1 WHERE edumpi_run_id = $2";
        const char *update_paramValues[2] = {end_time_str, job_id};
        int update_paramLengths[2] = {strlen(end_time_str), strlen(job_id)};
        int update_paramFormats[2] = {0, 0};

        // PostgreSQL Abfrage für das Update
        res = PQexecParams(conn, update_query, 2, NULL, update_paramValues, update_paramLengths, update_paramFormats, 0);
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            //fprintf(stderr, "UPDATE failed: %s", PQerrorMessage(conn));
        } 
        PQclear(res);
    }
    
    PQflush(conn);
    PQfinish(conn);
    //free(ringbuffer);
    //fwrite(buffer, offset, 1, file);
    //writeToPostgres(conn, buffer, offset);
}

struct timeval start_time, end_time;
float total_time;

qentry* getWritingRingPos(){

    if(amt_mode){
        return edumpi_amt_ring_claim(&amt_ring);
    }

    if(writer_pos == -1){
        writer_pos ++;
        edumpi_entry_reset(&ringbuffer[0], ++edumpi_ticket_counter);
        return &ringbuffer[0];
    }
    
    //Ringbuffer ist voll, es müssen erst Items ausgelesen werden
    //gettimeofday(&start_time, NULL);
    while(writer_pos == reader_pos-1 || (reader_pos == 0 && writer_pos == MAX_RINGSIZE)){
        edumpi_idle_sleep();
    }
    //gettimeofday(&end_time, NULL);
    //total_time += timeDifference(end_time, start_time);
    //Letzte Position im Ring-Buffer erreicht
    
    if(writer_pos == MAX_RINGSIZE-1){
        writer_pos = 0;
        edumpi_entry_reset(&ringbuffer[0], ++edumpi_ticket_counter);
    }
    //Solange Writer vor dem Reader kann nichts passieren
    else{
       qentry* item = &ringbuffer[writer_pos+1];
       edumpi_entry_reset(item, ++edumpi_ticket_counter);
       writer_pos++;
       //printf("writing\n");
    }
    //printf("Adressen-Test; %p\n", &ringbuffer[writer_pos]);
    return &ringbuffer[writer_pos];
}

char* createIntArray(int count){
    char buffer_help[8];
    int i = 0;
    while(count!=0){
        int rem = count%10;
        buffer_help[i++] = (rem>9)?(rem-10)+'a':rem+'0';
        count = count / 10;
    }
    buffer_help[i] = '\0';
    char *buffer_help2 = (char*) malloc(i-1);
    int x = 0;
    while(i>0){
        i--;
        buffer_help2[x] = buffer_help[i];
        x++;
    }
    buffer_help2[x] = '\0';
    return buffer_help2;
}


void closeFile(){
    //fwrite(PGCOPY_TRAILER, 2, 1, file);  
    //close(file);
    printf("Time: %f\n", total_time);
    printf("Counter: %d\n", counter);
}

void openFile(){
    char filename[20];
    sprintf(filename, "./data_rank_%d.bin", processrank);
    file = fopen(filename, "wb");
    fwrite(PGCOPY_HEADER, 19, 1, file);
}

void initializeQueue()
{ 
    //gettimeofday(&init_sql_start, NULL);

#ifdef SEPERATE_THREAD_NO_HYPERTHREADING
    hwloc_topology_t topology;
    hwloc_cpuset_t cpuset;
    pthread_t threads[1];  // Nur ein zusätzlicher Thread (der Logging-Thread)
    int core_ids[2];  // Array, um die zugeordneten Cores zu speichern
    pthread_attr_t attr_main, attr_logging; 

    // Topologie initialisieren
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // CPU-Set initialisieren
    cpuset = hwloc_bitmap_alloc();
    if (!cpuset) {
        fprintf(stderr, "Failed to allocate CPU set\n");
        hwloc_topology_destroy(topology);
        return;
    }

    // CPU-Bindung abrufen
    if (hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_PROCESS) < 0) {
        fprintf(stderr, "Failed to get CPU binding\n");
        hwloc_bitmap_free(cpuset);
        hwloc_topology_destroy(topology);
        return;
    }

    // CPU-Set in lesbaren String konvertieren
    char *str;
    hwloc_bitmap_asprintf(&str, cpuset);
    if (str) {
        printf("Assigned cores: %s\n", str);
        free(str);
    } else {
        fprintf(stderr, "Failed to convert CPU set to string\n");
    }
    
    main_core = hwloc_bitmap_first(cpuset);
    if(main_core == -1){
        fprintf(stderr, "No available cores found for main process\n");
        hwloc_bitmap_free(cpuset);
        hwloc_topology_destroy(topology);
        return;
    }
    
    core_ids[0] = main_core;
    
    logging_core = hwloc_bitmap_next(cpuset, main_core);
    if (logging_core == -1) {
        fprintf(stderr, "No available cores found for logging thread\n");
        hwloc_bitmap_free(cpuset);
        hwloc_topology_destroy(topology);
        return;
    }
    
    core_ids[1] = logging_core;
    
    pthread_attr_init(&attr_main);
    cpu_set_t cpu_set_main;
    
    CPU_ZERO(&cpu_set_main);
    CPU_SET(core_ids[0], &cpu_set_main);
    pthread_attr_setaffinity_np(&attr_main, sizeof(cpu_set_t), &cpu_set_main);
#endif
    
    TAILQ_INIT(&head);
    
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &processrank);
    MPI_Get_processor_name(proc_name, &proc_name_length);

#ifdef SEPERATE_THREAD_NO_HYPERTHREADING
    pthread_attr_init(&attr_logging);
    cpu_set_t cpu_set_logging;
    CPU_ZERO(&cpu_set_logging);
    CPU_SET(core_ids[1], &cpu_set_logging);
    pthread_attr_setaffinity_np(&attr_logging, sizeof(cpu_set_t), &cpu_set_logging);
    
    if(0 == pthread_create(&MONITOR_THREAD, &attr_logging, SQLMonitorFunc, &core_ids[1])){
        monitor_started = 1;
    }
    
    hwloc_bitmap_free(cpuset);
    hwloc_topology_destroy(topology);
#else
    if(0 == pthread_create(&MONITOR_THREAD, NULL, SQLMonitorFunc, NULL)){
        monitor_started = 1;
    }
#endif
    
    //gettimeofday(&init_sql_finished, NULL);
    //float dif = timeDifference(init_sql_finished, init_sql_start);
    //printf("Lost time for initializing sql: %f\n", dif);
}


/* Is the environment variable set to something other than 0/false/no? */
static int edumpi_env_enabled(const char *name){
    const char *v = getenv(name);
    if(v == NULL || *v == '\0'){
        return 0;
    }
    return !(0 == strcmp(v, "0") || 0 == strcasecmp(v, "false") || 0 == strcasecmp(v, "no"));
}

/*
 * Set up the event ring. Called before ompi_mpi_init() by MPI_Init and
 * MPI_Init_thread.
 *
 * EDUMPI_MPI_THREAD_AMT=1 selects the multi-producer ring for programs that call
 * MPI from many threads at once (MPI_THREAD_MULTIPLE, e.g. asynchronous
 * many-task runtimes). It is off by default. There is still one ring and one
 * monitor thread per rank. EDUMPI_AMT_STALL_MS (default 10000) is how long an
 * unfinished entry may hold back the entries behind it.
 *
 * EDUMPI_REQUEST_WAIT_MS (default 2000, both modes) is how long an entry of a
 * non-blocking receive waits for its request to finish, so that the number of
 * received bytes is in the database.
 */
void edumpi_analysis_init_pre(void)
{
    run_thread = 1;
    counter = 0;
    amt_mode = edumpi_env_enabled("EDUMPI_MPI_THREAD_AMT");
    edumpi_recv_duration = edumpi_env_enabled("EDUMPI_RECV_DURATION");
    edumpi_send_completion = amt_mode && edumpi_env_enabled("EDUMPI_SEND_COMPLETION");
    const char *wait = getenv("EDUMPI_REQUEST_WAIT_MS");
    int64_t request_wait_ms = (wait != NULL && atol(wait) >= 0) ? atol(wait) : 2000;
    request_wait_s = (double) request_wait_ms / 1000.0;
    if(amt_mode){
        const char *stall = getenv("EDUMPI_AMT_STALL_MS");
        int64_t stall_ms = (stall != NULL && atol(stall) > 0) ? atol(stall) : 10000;
        if(0 != edumpi_amt_ring_init(&amt_ring, MAX_RINGSIZE, stall_ms, request_wait_ms)){
            fprintf(stderr, "EduMPI: could not allocate the AMT ring buffer\n");
            abort();
        }
    } else {
        ringbuffer = (qentry*)malloc(sizeof(qentry)*MAX_RINGSIZE);
        writer_pos = -1;
        reader_pos = -1;
    }
    q_qentry = (qentry*)malloc(sizeof(qentry));
    job_id = getenv("SLURM_JOBID");
    clock_gettime(CLOCK_REALTIME, &start_runtime);
}

/* Start the monitor thread. Called after ompi_mpi_init(), as it needs the rank. */
void edumpi_analysis_init_post(int provided)
{
    initializeQueue();
    if(!amt_mode && provided == MPI_THREAD_MULTIPLE && processrank == 0){
        fprintf(stderr, "EduMPI: MPI_THREAD_MULTIPLE is active but EDUMPI_MPI_THREAD_AMT is not set. "
                        "If MPI is called from several threads at once, set EDUMPI_MPI_THREAD_AMT=1.\n");
    }
}

static const char FUNC_NAME[] = "MPI_Init";
int MPI_Init(int *argc, char ***argv)
{
#ifdef ENABLE_ANALYSIS
    edumpi_analysis_init_pre();
#endif

    //buffer = (char*)malloc(2000000);
    //offset = 0;
    #ifdef ENABLE_ANALYSIS
    gettimeofday(&start, NULL);
    #endif
    int err;
    int provided;
    int required = MPI_THREAD_SINGLE;

    /* check for environment overrides for required thread level.  If
       there is, check to see that it is a valid/supported thread level.
       If not, default to MPI_THREAD_MULTIPLE. */
    if (OMPI_SUCCESS > ompi_getenv_mpi_thread_level(&required)) {
        required = MPI_THREAD_MULTIPLE;
    }
    //printf("Test 2\n");

    /* Call the back-end initialization function (we need to put as
       little in this function as possible so that if it's profiled, we
       don't lose anything) */

    if (NULL != argc && NULL != argv) {
        err = ompi_mpi_init(*argc, *argv, required, &provided, false);
    } else {
        err = ompi_mpi_init(0, NULL, required, &provided, false);
    }
    //printf("Test 3\n");

    /* Since we don't have a communicator to invoke an errorhandler on
       here, don't use the fancy-schmancy ERRHANDLER macros; they're
       really designed for real communicator objects.  Just use the
       back-end function directly. */

    if (MPI_SUCCESS != err) {
        return ompi_errhandler_invoke(NULL, NULL,
                                      OMPI_ERRHANDLER_TYPE_COMM,
                                      err <
                                      0 ? ompi_errcode_get_mpi_code(err) :
                                      err, FUNC_NAME);
    }
    #ifdef ENABLE_ANALYSIS
    	edumpi_analysis_init_post(provided);
    #endif
    //printf("hier gehts weiter\n");
    SPC_INIT();
    return MPI_SUCCESS;
}
