/*
 * EduMPI: event entries of non-blocking requests, shared by the PMLs (UCX, ob1).
 *
 * The entry of an MPI_Irecv is written when the receive finishes, the entry of an MPI_Isend can get a second
 * entry "MPI_Isend_done" when the send request finishes. Each PML keeps a small area (edumpi_req_ext_t) in its
 * request and calls the functions below from its completion path. The PML only has to provide
 *   - the area in its request, reset when the request is created,
 *   - edumpi_req_attach_recv() / edumpi_req_attach_send() when MPI_Irecv / MPI_Isend returned,
 *   - edumpi_req_recv_finished() / edumpi_req_send_finished() before the request is completed.
 */
#ifndef EDUMPI_REQ_EXT_H
#define EDUMPI_REQ_EXT_H

#ifdef ENABLE_ANALYSIS
#include <time.h>
#include "ompi/mpi/c/init.h"

/*
 * Private area of a request. It links an MPI_Irecv request to its event entry, so that the completion path
 * can store the number of received bytes in the entry and tell the monitor thread that the entry may be
 * written.
 */
typedef struct edumpi_req_ext {
    qentry  *q;        /* entry of the MPI_Irecv, set after the receive was posted */
    uint64_t ticket;   /* ticket of that entry (see lifecycle in init.h)           */
    int64_t  bytes;    /* received bytes, valid if cb_done is set                  */
    int      source;   /* rank of the sender, valid if cb_done is set (-1: none)   */
    int      cb_done;  /* the completion path has run                              */
    int      finished; /* the entry was completed (in place or as a new entry)     */
    int      partner;  /* partner rank the receive was posted with                 */
    MPI_Comm communicator;
    char     area[64]; /* name of the communicator; with partner and communicator the snapshot
                        * to write a new entry if the monitor thread abandoned the old one */
    /* MPI_Isend with EDUMPI_SEND_COMPLETION: snapshot of the send, written as an "MPI_Isend_done"
     * entry when the send request completes */
    int      send_armed;    /* MPI_Isend has attached the snapshot                     */
    int      send_cb_done;  /* the completion path has run                             */
    int      send_finished; /* the "MPI_Isend_done" entry was written (only once)      */
    int      send_partner;
    struct timespec send_start; /* start of the MPI_Isend call */
    struct timespec send_done;  /* time the request completed  */
} edumpi_req_ext_t;

/* EDUMPI_SEND_COMPLETION=1 (AMT mode): defined in ompi/mpi/c/init.c */
extern int edumpi_send_completion;

static inline void edumpi_req_ext_reset(edumpi_req_ext_t *x)
{
    x->q = NULL;
    x->ticket = 0;
    x->bytes = 0;
    x->source = -1;
    x->cb_done = 0;
    x->finished = 0;
    x->partner = -1;
    x->communicator = NULL;
    x->area[0] = '\0';
    x->send_armed = 0;
    x->send_cb_done = 0;
    x->send_finished = 0;
    x->send_partner = -1;
}

/*
 * The receive finished: the entry gets the received size and, for a receive that was posted
 * with MPI_ANY_SOURCE, the rank of the sender (the entry has the wildcard until now). Then the
 * monitor thread may write the entry.
 *
 * Exactly one of two things happens, decided by one compare-and-swap on the lifecycle word of
 * the entry (the ticket makes it safe if the slot was reused):
 *  - the entry is still waiting: it is completed in place, and
 *  - the monitor thread gave up waiting (the receive was idle longer than
 *    EDUMPI_REQUEST_WAIT_MS) and freed the slot without writing it: a new entry with the data
 *    is written now, from the snapshot in the request.
 * In both cases nothing is lost and nothing is written twice.
 *
 * The entry is stamped with the time the message arrived, not with the time the receive was
 * posted. A receive that is posted early (HPX keeps a pool of them posted all the time) would
 * otherwise show up seconds before its message, in another second of the timeline than the
 * matching send.
 */
static inline void edumpi_req_complete_entry(edumpi_req_ext_t *x, qentry *q,
                                             int64_t bytes, int source)
{
    if (0 != __atomic_exchange_n(&x->finished, 1, __ATOMIC_SEQ_CST)) {
        return; /* the other side (completion path or MPI_Irecv) did it already */
    }

    uint64_t expected = EDUMPI_LC(x->ticket, EDUMPI_LC_PENDING);
    if (__atomic_compare_exchange_n(&q->lifecycle, &expected,
                                    EDUMPI_LC(x->ticket, EDUMPI_LC_FINISHING), 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        if (source >= 0 && q->partnerrank < 0) {
            q->partnerrank = edumpi_world_rank(q->communicator, source);
        }
        clock_gettime(CLOCK_REALTIME, &q->end);
        /* keep the time the receive was posted (the wrapper set start then): the row writer uses it for the
         * time_diff of the receive if EDUMPI_RECV_DURATION is set. start itself becomes the arrival time. */
        q->initializeRequest = q->start;
        q->start = q->end;
        edumpi_request_done_bytes(q, bytes);
        return;
    }

    /* abandoned: write a new, complete entry */
    qentry *n = getWritingRingPos();
    initQentry(&n, (source >= 0) ? edumpi_world_rank(x->communicator, source) : x->partner, "MPI_Irecv", 9, 0, 0, "p2p", 3,
               NULL, NULL, NULL, 0, NULL);
    memcpy(n->communicationArea, x->area, sizeof(n->communicationArea));
    n->communicator = x->communicator;
    n->recvDatasize = bytes;
    clock_gettime(CLOCK_REALTIME, &n->start);
    /* the monitor thread writes an entry when its end is set: the seconds come last */
    n->end.tv_nsec = n->start.tv_nsec;
    __atomic_store_n(&n->end.tv_sec, n->start.tv_sec, __ATOMIC_RELEASE);
}

/* Completion side of a receive. Call before ompi_request_complete(): afterwards the request may be
 * freed and reused. */
static inline void edumpi_req_recv_finished(edumpi_req_ext_t *x, int64_t received, int source)
{
    __atomic_store_n(&x->bytes, received, __ATOMIC_RELAXED);
    __atomic_store_n(&x->source, source, __ATOMIC_RELAXED);
    __atomic_store_n(&x->cb_done, 1, __ATOMIC_SEQ_CST);
    qentry *q = __atomic_load_n(&x->q, __ATOMIC_SEQ_CST);
    if (NULL != q) {
        edumpi_req_complete_entry(x, q, received, source);
    }
}

/* MPI_Irecv side. The receive may already be finished (the completion can run before this), so
 * check again after the entry was attached. At least one of the two sides sees the other one; the
 * entry is completed only once. */
static inline void edumpi_req_attach_recv(edumpi_req_ext_t *x, qentry *q)
{
    /* snapshot for a new entry, in case the monitor thread abandons this one */
    x->ticket = __atomic_load_n(&q->lifecycle, __ATOMIC_RELAXED) >> 2;
    x->partner = q->partnerrank;
    x->communicator = q->communicator;
    memcpy(x->area, q->communicationArea, sizeof(x->area));
    /* from now on the monitor thread may abandon the entry (it does so only after the
     * MPI_Irecv call returned) */
    __atomic_store_n(&q->lifecycle, EDUMPI_LC(x->ticket, EDUMPI_LC_PENDING), __ATOMIC_RELEASE);
    __atomic_store_n(&x->q, q, __ATOMIC_SEQ_CST);
    if (0 != __atomic_load_n(&x->cb_done, __ATOMIC_SEQ_CST)) {
        edumpi_req_complete_entry(x, q, x->bytes, x->source);
    }
}

/* Is the entry of this receive still the one of this request (not abandoned, slot not reused)?
 * For the PML to update fields of the entry (late sender time ...) while the receive is in flight. */
static inline qentry *edumpi_req_live_entry(edumpi_req_ext_t *x)
{
    qentry *q = __atomic_load_n(&x->q, __ATOMIC_SEQ_CST);
    if (NULL == q) {
        return NULL;
    }
    return (__atomic_load_n(&q->lifecycle, __ATOMIC_ACQUIRE) == EDUMPI_LC(x->ticket, EDUMPI_LC_PENDING)) ? q : NULL;
}

/*
 * Completion of a send request that was still in flight when MPI_Isend returned (a message that is not
 * sent eagerly). The entry of MPI_Isend only knows the length of the call. When the request completes, a
 * second, small entry "MPI_Isend_done" is written: its end is the completion time and its time_diff
 * (completion minus the start of the MPI_Isend call) is the lifetime of the send request, so that the
 * analysis can show how long the send really took. Both entries have the same call start, which pairs them.
 * A new entry is used (as for an abandoned MPI_Irecv) so that nothing is written into an entry that the
 * monitor thread may already have written and freed.
 */
static inline void edumpi_req_send_write_done(edumpi_req_ext_t *x)
{
    if (0 != __atomic_exchange_n(&x->send_finished, 1, __ATOMIC_SEQ_CST)) {
        return; /* the other side (completion path or MPI_Isend) did it already */
    }
    qentry *n = getWritingRingPos();
    initQentry(&n, x->send_partner, "MPI_Isend_done", 14, 0, 0, "p2p", 3, NULL, NULL, NULL, 0, NULL);
    memcpy(n->communicationArea, x->area, sizeof(n->communicationArea));
    n->communicator = x->communicator;
    n->start = x->send_start;
    /* the monitor thread writes an entry when its end is set: the seconds come last */
    n->end.tv_nsec = x->send_done.tv_nsec;
    __atomic_store_n(&n->end.tv_sec, x->send_done.tv_sec, __ATOMIC_RELEASE);
}

/* Completion side of a send. Call at the start of the send completion. */
static inline void edumpi_req_send_finished(edumpi_req_ext_t *x)
{
    clock_gettime(CLOCK_REALTIME, &x->send_done);
    __atomic_store_n(&x->send_cb_done, 1, __ATOMIC_SEQ_CST);
    if (0 != __atomic_load_n(&x->send_armed, __ATOMIC_SEQ_CST)) {
        edumpi_req_send_write_done(x);
    }
}

/* MPI_Isend side. The request may already be complete (the completion can run before this), so check
 * again after the snapshot was attached. At least one of the two sides sees the other one. */
static inline void edumpi_req_attach_send(edumpi_req_ext_t *x, qentry *q, MPI_Comm comm)
{
    x->send_partner = q->partnerrank;
    x->communicator = comm;
    memcpy(x->area, q->communicationArea, sizeof(x->area));
    x->send_start = q->start;
    __atomic_store_n(&x->send_armed, 1, __ATOMIC_SEQ_CST);
    if (0 != __atomic_load_n(&x->send_cb_done, __ATOMIC_SEQ_CST)) {
        edumpi_req_send_write_done(x);
    }
}

#endif /* ENABLE_ANALYSIS */
#endif /* EDUMPI_REQ_EXT_H */
