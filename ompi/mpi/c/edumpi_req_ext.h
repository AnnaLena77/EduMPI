/*
 * EduHPX measuring MPI: entries of non-blocking requests, shared by the PMLs (UCX, ob1).
 *
 * The entry of an MPI_Irecv gets the time the message arrived ("done"), the number of received bytes and, for
 * MPI_ANY_SOURCE, the real sender when the receive completes. The entry of an MPI_Isend that is still in flight when
 * the call returns gets the time its request completed ("done"). Each PML keeps a small area (edumpi_req_ext_t) in
 * its request and calls the functions below from its completion path. The PML only has to provide
 *   - the area in its request, reset when the request is created,
 *   - edumpi_req_attach_recv() / edumpi_req_attach_send() when MPI_Irecv / MPI_Isend returned,
 *   - edumpi_req_recv_finished() / edumpi_req_send_finished() before the request is completed.
 *
 * The entry lives in a segment of the event store that never moves and is not written to the log before the request
 * completed (pending_request / request_done in the entry), so the completion can write into it by pointer at any time.
 * Completion and attach can happen in either order (on different threads): both sides check, one of them finishes.
 */
#ifndef EDUMPI_REQ_EXT_H
#define EDUMPI_REQ_EXT_H

#ifdef ENABLE_ANALYSIS
#include <time.h>
#include "ompi/mpi/c/init.h"

typedef struct edumpi_req_ext {
    qentry  *q;        /* entry of the MPI_Irecv / MPI_Isend, set after the call posted the request */
    int64_t  bytes;    /* received bytes, valid if cb_done is set                  */
    int      source;   /* rank of the sender, valid if cb_done is set (-1: none)   */
    int      cb_done;  /* the receive completion has run                           */
    int      finished; /* the entry of the receive was completed                   */
    struct timespec done;       /* arrival time of the message                      */
    int      send_armed;        /* MPI_Isend has attached its entry                 */
    int      send_cb_done;      /* the send completion has run                      */
    int      send_finished;     /* the entry of the send was completed (once)       */
    struct timespec send_done;  /* time the send request completed                  */
} edumpi_req_ext_t;

/* defined in ompi/mpi/c/init.c */
extern int edumpi_send_completion;

static inline void edumpi_req_ext_reset(edumpi_req_ext_t *x)
{
    x->q = NULL;
    x->bytes = 0;
    x->source = -1;
    x->cb_done = 0;
    x->finished = 0;
    x->send_armed = 0;
    x->send_cb_done = 0;
    x->send_finished = 0;
}

/* The receive finished and the entry is known: store size, real source and arrival time, then the entry may be
 * written. Done once, by whichever side comes second. */
static inline void edumpi_req_complete_entry(edumpi_req_ext_t *x, qentry *q, int64_t bytes, int source)
{
    if (0 != __atomic_exchange_n(&x->finished, 1, __ATOMIC_SEQ_CST)) {
        return;
    }
    if (source >= 0 && q->partnerrank < 0) {
        q->partnerrank = edumpi_world_rank(q->communicator, source);
        q->recv_any_source = 1;
    }
    q->done = x->done;
    edumpi_request_done_bytes(q, bytes);
}

/* Completion side of a receive. Call before ompi_request_complete(): afterwards the request may be freed and reused. */
static inline void edumpi_req_recv_finished(edumpi_req_ext_t *x, int64_t received, int source)
{
    clock_gettime(CLOCK_REALTIME, &x->done);
    __atomic_store_n(&x->bytes, received, __ATOMIC_RELAXED);
    __atomic_store_n(&x->source, source, __ATOMIC_RELAXED);
    __atomic_store_n(&x->cb_done, 1, __ATOMIC_SEQ_CST);
    qentry *q = __atomic_load_n(&x->q, __ATOMIC_SEQ_CST);
    if (NULL != q) {
        edumpi_req_complete_entry(x, q, received, source);
    }
}

/* MPI_Irecv side: the receive may already be finished, so check again after the entry was attached. */
static inline void edumpi_req_attach_recv(edumpi_req_ext_t *x, qentry *q)
{
    __atomic_store_n(&x->q, q, __ATOMIC_SEQ_CST);
    if (0 != __atomic_load_n(&x->cb_done, __ATOMIC_SEQ_CST)) {
        edumpi_req_complete_entry(x, q, x->bytes, x->source);
    }
}

static inline void edumpi_req_complete_send(edumpi_req_ext_t *x, qentry *q)
{
    if (0 != __atomic_exchange_n(&x->send_finished, 1, __ATOMIC_SEQ_CST)) {
        return;
    }
    q->done = x->send_done;
    edumpi_request_done(q);
}

/* Completion side of a send. Call at the start of the send completion. */
static inline void edumpi_req_send_finished(edumpi_req_ext_t *x)
{
    clock_gettime(CLOCK_REALTIME, &x->send_done);
    __atomic_store_n(&x->send_cb_done, 1, __ATOMIC_SEQ_CST);
    if (0 != __atomic_load_n(&x->send_armed, __ATOMIC_SEQ_CST)) {
        edumpi_req_complete_send(x, __atomic_load_n(&x->q, __ATOMIC_SEQ_CST));
    }
}

/* MPI_Isend side (the send is still in flight): the entry waits for the completion. The request may already be
 * complete, so check again after the entry was attached. */
static inline void edumpi_req_attach_send(edumpi_req_ext_t *x, qentry *q, MPI_Comm comm)
{
    (void) comm;
    __atomic_store_n(&q->pending_request, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&x->q, q, __ATOMIC_SEQ_CST);
    __atomic_store_n(&x->send_armed, 1, __ATOMIC_SEQ_CST);
    if (0 != __atomic_load_n(&x->send_cb_done, __ATOMIC_SEQ_CST)) {
        edumpi_req_complete_send(x, q);
    }
}

#endif /* ENABLE_ANALYSIS */
#endif /* EDUMPI_REQ_EXT_H */
