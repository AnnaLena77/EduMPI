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
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <sched.h>
#include <math.h>
#include <sys/time.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "opal/util/show_help.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/mpi/c/bindings.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/constants.h"

#ifdef ENABLE_ANALYSIS
#include "ompi/mpi/c/init.h"
#include "ompi/peruse/peruse.h"
#endif


#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Init = PMPI_Init
#endif
#define MPI_Init PMPI_Init
#endif

static struct timeval start;
static struct timespec start_runtime;

static int size, processrank;
char proc_name[MPI_MAX_PROCESSOR_NAME];
int proc_name_length;

qentry *q_qentry;

/* Hooks of the PML (pml_ucx_request.h, ob1): a send that is still in flight when MPI_Isend returns gets the time its
 * request completed. Read by the PMLs, therefore not static. Default on, EDUHPX_MPI_SEND_COMPLETION=0 turns it off. */
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

static int edumpi_comm_id(MPI_Comm comm);

void initQentry(qentry **q, int dest, char *function, int function_len, int sendCount, int recvCount, char *commType, int commType_len, MPI_Datatype sendType, MPI_Datatype recvType, MPI_Comm comm, int blocking, MPI_Op op){
    if(q==NULL || *q==NULL){
    	return;
    } else {
        qentry *item = *q;                /* a new entry is zeroed memory: only what is not zero is stored */
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
        
        item->comm_id = edumpi_comm_id(comm) + 1;     /* the label is looked up once per communicator, not written into every entry */
        if(comm != NULL){
            item->communicator = comm;
        }
        item->processrank = processrank;
        item->partnerrank = edumpi_world_rank(comm, dest);
        
        //printf("Test Rank: %d, WriterPos: %d, Bcast: %p\n", processrank, writer_pos, item);
        
       
        memcpy(item->sendmode, "", 0);
        memcpy(item->usedBtl, "", 0);
        memcpy(item->usedProtocol, "", 0);
        item->withinEagerLimit = -1;
        item->foundMatchWild = -1;
        memcpy(item->usedAlgorithm, "", 0);

    }
}


/*
 * EduHPX measuring MPI: event store and log file.
 *
 * Every MPI call claims an entry (qentry) from a lock-free store and fills it in, as it always did. The store is a
 * table of segments of EDUHPX_SEG_EVENTS entries each. A segment is anonymous memory that never moves, so the
 * completion hooks of the PMLs can write into the entry of their request by pointer, however late the request
 * completes. There is no monitor thread and no database: when a new segment is started, the calling thread
 * converts the older segments whose entries are all complete into 64-byte records and appends them to
 * eduhpx.mpi.<rank>.bin; MPI_Finalize writes the rest. A segment that still holds an entry whose call has not
 * returned or whose request has not completed (a receive that is posted and has not got its message) stays in memory
 * until it is complete or until MPI_Finalize. The format is described in docs/eduhpx_mpi_log_format.md.
 */
#define EDUHPX_SEG_EVENTS 2048
#define EDUHPX_SEG_BYTES  ((size_t) EDUHPX_SEG_EVENTS * sizeof(qentry))
#define EDUHPX_MAX_SEGS   (1 << 18)
#define EDUHPX_REC        64
#define EDUHPX_HDR        64
#define EDUHPX_RETRY_SEGS 8            /* look at a segment that was not complete again after this many new segments */
#define EDUHPX_QUIET_NS   100000       /* an entry counts as complete only 100 us after its end was set (no half-written stamp) */
#define EDUHPX_MAX_FUNCS  512
#define EDUHPX_MAX_COMMS  4096
#define EDUHPX_SEG_FLUSHED ((void *) 1)

enum {
    EDUHPX_F_BLOCKING      = 1u << 0, /* blocking call */
    EDUHPX_F_COLLECTIVE    = 1u << 1, /* collective, not point-to-point */
    EDUHPX_F_NO_END        = 1u << 2, /* the call had not returned at MPI_Finalize */
    EDUHPX_F_HAS_DONE      = 1u << 3, /* done_ns is valid (MPI_Irecv: arrival, MPI_Isend: completion of the request) */
    EDUHPX_F_ANY_SOURCE    = 1u << 4, /* receive posted with MPI_ANY_SOURCE, partner is the real sender */
    EDUHPX_F_NOT_COMPLETED = 1u << 5  /* the request had not completed at MPI_Finalize */
};

static void *edumpi_seg_table[EDUHPX_MAX_SEGS];
static uint32_t edumpi_seg_retry[EDUHPX_MAX_SEGS];
static uint64_t edumpi_next_event = 0;     /* events claimed */
static uint64_t edumpi_scan_from = 0;      /* first segment that is not flushed */
static int edumpi_flush_busy = 0;
static uint64_t edumpi_dropped = 0;
static uint64_t edumpi_records = 0;
static int edumpi_log_fd = -1;
static int edumpi_finalized = 0;
static char edumpi_log_dir[1024];
static char edumpi_func_names[EDUHPX_MAX_FUNCS][32];
static int edumpi_nfunc = 0;
static char edumpi_comm_names[EDUHPX_MAX_COMMS][64];
static int edumpi_ncomm = 0;
static qentry edumpi_scratch_entry;        /* for events that cannot be stored (store full) */

static void edumpi_put16(unsigned char *p, uint16_t v){ p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; }
static void edumpi_put32(unsigned char *p, uint32_t v){ for(int i = 0; i < 4; i++){ p[i] = (v >> (8 * i)) & 0xff; } }
static void edumpi_put64(unsigned char *p, uint64_t v){ for(int i = 0; i < 8; i++){ p[i] = (v >> (8 * i)) & 0xff; } }

static uint64_t edumpi_ts_ns(struct timespec t){
    if(t.tv_sec == 0 && t.tv_nsec == 0){
        return 0;
    }
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

static uint64_t edumpi_now_ns(void){
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

/* index of name in a table of fixed-size names (added if new); -1 if the table is full */
static int edumpi_intern(char *table, int *count, int max, size_t width, const char *name, size_t maxlen){
    size_t len = strnlen(name, maxlen);
    if(len >= width){
        len = width - 1;
    }
    for(int i = *count - 1; i >= 0; i--){
        const char *e = table + (size_t) i * width;
        if(0 == strncmp(e, name, len) && e[len] == '\0'){
            return i;
        }
    }
    if(*count >= max){
        return -1;
    }
    char *slot = table + (size_t) (*count) * width;
    memset(slot, 0, width);
    memcpy(slot, name, len);
    return (*count)++;
}

/* Index in the table of communicator labels of this communicator (-1: none). A small cache, so that the label is built
 * (snprintf) once per communicator and not at every call. */
#define EDUMPI_COMM_CACHE 32
static struct { MPI_Comm comm; int cindex; int id; } edumpi_comm_cache[EDUMPI_COMM_CACHE];
static int edumpi_comm_cache_n = 0;
static int edumpi_comm_world_id = -2;
static int edumpi_comm_lock = 0;
static int edumpi_comm_id(MPI_Comm comm){
    if(comm == NULL || comm == MPI_COMM_NULL){
        return -1;
    }
    int world = (comm == MPI_COMM_WORLD);
    if(world){
        int id = __atomic_load_n(&edumpi_comm_world_id, __ATOMIC_ACQUIRE);
        if(id != -2){
            return id;
        }
    } else {
        int n = __atomic_load_n(&edumpi_comm_cache_n, __ATOMIC_ACQUIRE);
        for(int i = 0; i < n; i++){
            if(edumpi_comm_cache[i].comm == comm && edumpi_comm_cache[i].cindex == (int) comm->c_index){
                return edumpi_comm_cache[i].id;
            }
        }
    }
    while(__atomic_exchange_n(&edumpi_comm_lock, 1, __ATOMIC_ACQUIRE)){ }
    char label[64];
    edumpi_comm_label(comm, label, sizeof(label));
    int id = (label[0] == '\0') ? -1 : edumpi_intern(&edumpi_comm_names[0][0], &edumpi_ncomm, EDUHPX_MAX_COMMS, 64, label, sizeof(label));
    if(world){
        __atomic_store_n(&edumpi_comm_world_id, id, __ATOMIC_RELEASE);
    } else if(edumpi_comm_cache_n < EDUMPI_COMM_CACHE){
        edumpi_comm_cache[edumpi_comm_cache_n].comm = comm;
        edumpi_comm_cache[edumpi_comm_cache_n].cindex = (int) comm->c_index;
        edumpi_comm_cache[edumpi_comm_cache_n].id = id;
        __atomic_store_n(&edumpi_comm_cache_n, edumpi_comm_cache_n + 1, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&edumpi_comm_lock, 0, __ATOMIC_RELEASE);
    return id;
}

/* The entry is final: the call returned, its request (if it has one) completed. */
static int edumpi_entry_complete(qentry *e, uint64_t now_ns){
    if(e->end.tv_sec == 0 && e->end.tv_nsec == 0){
        return 0;
    }
    if(!edumpi_request_ready(e)){
        return 0;
    }
    uint64_t end = edumpi_ts_ns(e->end);
    return now_ns > end + EDUHPX_QUIET_NS;
}

static void edumpi_encode(qentry *e, uint64_t seq, unsigned char *r){
    uint32_t flags = 0;
    memset(r, 0, EDUHPX_REC);
    if(e->blocking){
        flags |= EDUHPX_F_BLOCKING;
    }
    if(0 == strncmp(e->communicationType, "collective", sizeof(e->communicationType))){
        flags |= EDUHPX_F_COLLECTIVE;
    }
    if(e->end.tv_sec == 0 && e->end.tv_nsec == 0){
        flags |= EDUHPX_F_NO_END;
    }
    if(!edumpi_request_ready(e)){
        flags |= EDUHPX_F_NOT_COMPLETED;
    }
    if(e->done.tv_sec != 0 || e->done.tv_nsec != 0){
        flags |= EDUHPX_F_HAS_DONE;
    }
    if(e->recv_any_source){
        flags |= EDUHPX_F_ANY_SOURCE;
    }
    int fn = edumpi_intern(&edumpi_func_names[0][0], &edumpi_nfunc, EDUHPX_MAX_FUNCS, 32, e->function, sizeof(e->function));
    int cm = (e->comm_id > 0) ? e->comm_id - 1 : 0xffff;
    if(flags & EDUHPX_F_COLLECTIVE){
        /* the done time and the "real source" of a collective are those of its last internal message: not exported */
        flags &= ~(EDUHPX_F_HAS_DONE | EDUHPX_F_ANY_SOURCE);
    }
    edumpi_put64(r + 0, edumpi_ts_ns(e->start));
    edumpi_put64(r + 8, edumpi_ts_ns(e->end));
    edumpi_put64(r + 16, (flags & EDUHPX_F_HAS_DONE) ? edumpi_ts_ns(e->done) : 0);
    edumpi_put64(r + 24, (uint64_t) e->sendDatasize);
    edumpi_put64(r + 32, (uint64_t) e->recvDatasize);
    edumpi_put32(r + 40, (uint32_t) e->partnerrank);
    edumpi_put16(r + 44, (uint16_t) (cm < 0 ? 0xffff : cm));
    edumpi_put16(r + 46, (uint16_t) (fn < 0 ? 0xffff : fn));
    edumpi_put32(r + 48, flags);
    edumpi_put32(r + 52, (uint32_t) e->sendcount);
    edumpi_put32(r + 56, (uint32_t) e->recvcount);
    edumpi_put32(r + 60, (uint32_t) seq);
}

static void edumpi_write_all(const void *buf, size_t len){
    const char *p = (const char *) buf;
    while(len > 0 && edumpi_log_fd >= 0){
        ssize_t n = write(edumpi_log_fd, p, len);
        if(n < 0){
            if(errno == EINTR){
                continue;
            }
            fprintf(stderr, "EduHPX MPI log: write failed: %s (further events are not written)\n", strerror(errno));
            close(edumpi_log_fd);
            edumpi_log_fd = -1;
            return;
        }
        p += n;
        len -= (size_t) n;
    }
}

/* Convert the entries of segment s to records and append them. With release the memory is given back (the address
 * stays valid and reads as zero, so that a late write of a PML into an entry does no harm). */
/* Event segments are expensive when they come new from the kernel (the pages are cleared and populated: about 0.5 us per event).
 * So a few are populated at start (MPI_Init) and flushed segments are kept for the next ones instead of being given back. */
#define EDUHPX_FREE_SEGS 16
#define EDUHPX_PREPOP_SEGS 8
static void *edumpi_free_segs[EDUHPX_FREE_SEGS];      /* flushed segments: must be cleared before they are used again */
static int edumpi_n_free = 0;
static void *edumpi_fresh_segs[EDUHPX_PREPOP_SEGS];   /* populated at start and still zero */
static int edumpi_n_fresh = 0;
static int edumpi_free_lock = 0;
static void edumpi_prepopulate(void){
    for(int i = 0; i < EDUHPX_PREPOP_SEGS; i++){
#ifdef MAP_POPULATE
        void *p = mmap(NULL, EDUHPX_SEG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_POPULATE, -1, 0);
#else
        void *p = mmap(NULL, EDUHPX_SEG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
        if(p == MAP_FAILED){
            break;
        }
        edumpi_fresh_segs[edumpi_n_fresh++] = p;
    }
}
static void *edumpi_fresh_pop(void){
    void *p = NULL;
    while(__atomic_exchange_n(&edumpi_free_lock, 1, __ATOMIC_ACQUIRE)){ }
    if(edumpi_n_fresh > 0){
        p = edumpi_fresh_segs[--edumpi_n_fresh];
    }
    __atomic_store_n(&edumpi_free_lock, 0, __ATOMIC_RELEASE);
    return p;
}
static void *edumpi_free_pop(void){
    void *p = NULL;
    while(__atomic_exchange_n(&edumpi_free_lock, 1, __ATOMIC_ACQUIRE)){ }
    if(edumpi_n_free > 0){
        p = edumpi_free_segs[--edumpi_n_free];
    }
    __atomic_store_n(&edumpi_free_lock, 0, __ATOMIC_RELEASE);
    return p;
}
static int edumpi_free_push(void *p){
    int ok = 0;
    while(__atomic_exchange_n(&edumpi_free_lock, 1, __ATOMIC_ACQUIRE)){ }
    if(edumpi_n_free < EDUHPX_FREE_SEGS){
        edumpi_free_segs[edumpi_n_free++] = p;
        ok = 1;
    }
    __atomic_store_n(&edumpi_free_lock, 0, __ATOMIC_RELEASE);
    return ok;
}

static void edumpi_flush_segment(uint64_t s, int release){
    void *base = edumpi_seg_table[s];
    uint64_t first = s * EDUHPX_SEG_EVENTS;
    uint64_t claimed = __atomic_load_n(&edumpi_next_event, __ATOMIC_ACQUIRE);
    uint64_t n = (claimed > first + EDUHPX_SEG_EVENTS) ? EDUHPX_SEG_EVENTS : (claimed > first ? claimed - first : 0);
    unsigned char *buf = (unsigned char *) malloc((size_t) EDUHPX_SEG_EVENTS * EDUHPX_REC);
    if(buf != NULL){
        for(uint64_t i = 0; i < n; i++){
            edumpi_encode((qentry *) base + i, first + i, buf + i * EDUHPX_REC);
        }
        edumpi_write_all(buf, (size_t) (n * EDUHPX_REC));
        edumpi_records += n;
        free(buf);
    }
    if(release && !edumpi_free_push(base)){
        mmap(base, EDUHPX_SEG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    }
    edumpi_seg_table[s] = EDUHPX_SEG_FLUSHED;
}

/* Flush the finished segments below "upto" (a thread that starts a new segment), or everything (force, at the end). */
static void edumpi_flush_ready(uint64_t upto, int force){
    if(__atomic_exchange_n(&edumpi_flush_busy, 1, __ATOMIC_ACQUIRE)){
        if(!force){
            return;
        }
        while(__atomic_exchange_n(&edumpi_flush_busy, 1, __ATOMIC_ACQUIRE)){
            sched_yield();
        }
    }
    uint64_t now = edumpi_now_ns();
    for(uint64_t s = edumpi_scan_from; s < upto && s < EDUHPX_MAX_SEGS; s++){
        void *base = edumpi_seg_table[s];
        if(base == NULL || base == EDUHPX_SEG_FLUSHED){
            continue;
        }
        if(!force){
            if(edumpi_seg_retry[s] > upto){
                continue;
            }
            int complete = 1;
            for(int i = 0; i < EDUHPX_SEG_EVENTS && complete; i++){
                complete = edumpi_entry_complete((qentry *) base + i, now);
            }
            if(!complete){
                edumpi_seg_retry[s] = (uint32_t) (upto + EDUHPX_RETRY_SEGS);
                continue;
            }
        }
        edumpi_flush_segment(s, !force);
    }
    while(edumpi_scan_from < EDUHPX_MAX_SEGS && edumpi_seg_table[edumpi_scan_from] == EDUHPX_SEG_FLUSHED){
        edumpi_scan_from++;
    }
    __atomic_store_n(&edumpi_flush_busy, 0, __ATOMIC_RELEASE);
}

qentry* getWritingRingPos(void){
    uint64_t idx = __atomic_fetch_add(&edumpi_next_event, 1, __ATOMIC_ACQ_REL);
    uint64_t seg = idx / EDUHPX_SEG_EVENTS;
    if(seg >= EDUHPX_MAX_SEGS){
        __atomic_fetch_add(&edumpi_dropped, 1, __ATOMIC_RELAXED);
        return &edumpi_scratch_entry;
    }
    void *base = __atomic_load_n(&edumpi_seg_table[seg], __ATOMIC_ACQUIRE);
    if(base == NULL){
        void *fresh = edumpi_fresh_pop();
        if(fresh == NULL){
            fresh = edumpi_free_pop();
            if(fresh != NULL){
                memset(fresh, 0, EDUHPX_SEG_BYTES);  /* a flushed segment: all entries start zeroed, as in new memory */
            }
        }
        if(fresh == NULL){
#ifdef MAP_POPULATE
            /* fault in the whole segment with one call instead of one page fault per few events */
            fresh = mmap(NULL, EDUHPX_SEG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_POPULATE, -1, 0);
#else
            fresh = mmap(NULL, EDUHPX_SEG_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#endif
        }
        if(fresh == MAP_FAILED){
            __atomic_fetch_add(&edumpi_dropped, 1, __ATOMIC_RELAXED);
            return &edumpi_scratch_entry;
        }
        void *expected = NULL;
        if(__atomic_compare_exchange_n(&edumpi_seg_table[seg], &expected, fresh, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)){
            base = fresh;
            if(seg > 0 && !edumpi_finalized){
                edumpi_flush_ready(seg, 0); /* the thread that fills the store pays for emptying it */
            }
        } else {
            if(!edumpi_free_push(fresh)){
                munmap(fresh, EDUHPX_SEG_BYTES);
            }
            base = expected;
        }
    }
    if(base == EDUHPX_SEG_FLUSHED){ /* cannot happen: only segments below the newest one are flushed, and all their events are claimed */
        __atomic_fetch_add(&edumpi_dropped, 1, __ATOMIC_RELAXED);
        return &edumpi_scratch_entry;
    }
    return (qentry *) base + (idx % EDUHPX_SEG_EVENTS);
}

void qentryIntoQueue(qentry **q){
    (void) q; /* the entry is already in the store since it was claimed */
}

/* Is the environment variable set to something other than 0/false/no? */
static int edumpi_env_enabled(const char *name){
    const char *v = getenv(name);
    if(v == NULL || *v == '\0'){
        return 0;
    }
    return !(0 == strcmp(v, "0") || 0 == strcasecmp(v, "false") || 0 == strcasecmp(v, "no"));
}

/* Before ompi_mpi_init(): nothing to set up except the switches (the store is static). */
void edumpi_analysis_init_pre(void)
{
    q_qentry = (qentry*)malloc(sizeof(qentry));
    /* the hooks of the PMLs: a send that is still in flight when MPI_Isend returns gets its completion time (the
     * "done" time of the entry). EDUHPX_MPI_SEND_COMPLETION=0 turns this off. */
    const char *sc = getenv("EDUHPX_MPI_SEND_COMPLETION");
    edumpi_send_completion = (sc == NULL) ? 1 : edumpi_env_enabled("EDUHPX_MPI_SEND_COMPLETION");
    clock_gettime(CLOCK_REALTIME, &start_runtime);
    edumpi_prepopulate();
}

/* Open the log file and write the header. Called after ompi_mpi_init(), as it needs the rank. */
void edumpi_analysis_init_post(int provided)
{
    (void) provided;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &processrank);
    MPI_Get_processor_name(proc_name, &proc_name_length);

    const char *dir = getenv("EDUHPX_MPI_LOG_DIR");
    snprintf(edumpi_log_dir, sizeof(edumpi_log_dir), "%s", (dir != NULL && *dir != '\0') ? dir : ".");
    char path[1200];
    snprintf(path, sizeof(path), "%s/eduhpx.mpi.%d.bin", edumpi_log_dir, processrank);
    edumpi_log_fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(edumpi_log_fd < 0){
        fprintf(stderr, "EduHPX MPI log: cannot open %s: %s (events are not written)\n", path, strerror(errno));
        return;
    }
    unsigned char hdr[EDUHPX_HDR];
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, "EDUHPXM1", 8);
    edumpi_put32(hdr + 8, 1);                 /* version */
    edumpi_put32(hdr + 12, EDUHPX_REC);       /* record size */
    edumpi_put32(hdr + 16, (uint32_t) processrank);
    edumpi_put32(hdr + 20, (uint32_t) size);
    snprintf((char *) hdr + 24, 32, "%s", proc_name);
    /* hdr + 56: offset of the trailer, written at the end (0 while the file is open) */
    edumpi_write_all(hdr, sizeof(hdr));
    atexit(edumpi_analysis_finalize); /* a program that does not reach MPI_Finalize still leaves a readable file */
}

/* MPI_Finalize: write everything that is left, then the trailer with the names, and put its offset in the header. */
void edumpi_analysis_finalize(void)
{
    if(__atomic_exchange_n(&edumpi_finalized, 1, __ATOMIC_ACQ_REL)){
        return;
    }
    uint64_t claimed = __atomic_load_n(&edumpi_next_event, __ATOMIC_ACQUIRE);
    uint64_t segs = (claimed + EDUHPX_SEG_EVENTS - 1) / EDUHPX_SEG_EVENTS;
    edumpi_flush_ready(segs, 1);
    if(edumpi_log_fd >= 0){
        off_t trailer_at = lseek(edumpi_log_fd, 0, SEEK_CUR);
        size_t tlen = 16 + (size_t) edumpi_nfunc * 32 + (size_t) edumpi_ncomm * 64;
        unsigned char *t = (unsigned char *) calloc(1, tlen);
        if(t != NULL){
            memcpy(t, "EDUHPXT1", 8);
            edumpi_put32(t + 8, (uint32_t) edumpi_nfunc);
            edumpi_put32(t + 12, (uint32_t) edumpi_ncomm);
            memcpy(t + 16, edumpi_func_names, (size_t) edumpi_nfunc * 32);
            memcpy(t + 16 + (size_t) edumpi_nfunc * 32, edumpi_comm_names, (size_t) edumpi_ncomm * 64);
            edumpi_write_all(t, tlen);
            free(t);
            unsigned char off[8];
            edumpi_put64(off, (uint64_t) trailer_at);
            if(pwrite(edumpi_log_fd, off, 8, 56) != 8){
                fprintf(stderr, "EduHPX MPI log: could not write the trailer offset\n");
            }
        }
        close(edumpi_log_fd);
        edumpi_log_fd = -1;
    }
    if(edumpi_dropped > 0){
        fprintf(stderr, "EduHPX MPI log (rank %d): %llu events were dropped because the store was full\n",
                processrank, (unsigned long long) edumpi_dropped);
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
