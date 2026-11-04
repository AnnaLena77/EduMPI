/*
 * EduMPI: multi-producer / single-consumer event ring for programs that call
 * MPI from many threads at the same time (MPI_THREAD_MULTIPLE), for example
 * asynchronous many-task (AMT) runtimes.
 *
 * It is used instead of the plain single-producer ring in init.c when the
 * environment variable EDUMPI_MPI_THREAD_AMT=1 is set.
 *
 * There is one ring and one consumer (the monitor thread) per MPI rank. Any
 * number of threads may claim slots concurrently:
 *
 *   FREE --producer: head CAS, zero the slot--> ACTIVE --consumer--> FREE
 *
 *  - A producer takes the next ticket (compare-and-swap on the head). The slot
 *    of that ticket is free as long as head - tail < size, which is checked
 *    before the ticket is taken. A producer never blocks and never spins: if
 *    the ring is full, the event is written to a scratch entry and counted as
 *    dropped, so tracing can never stall or deadlock the traced program.
 *  - The consumer reads the slots in claim order. A slot is complete when its
 *    "end" timestamp is set (this is what the MPI wrappers do when the call
 *    returns, as in the single-producer mode). An entry that stays incomplete
 *    longer than the stall timeout is emitted as it is, so one long blocking
 *    call cannot hold back the tracing of everything behind it.
 *  - An entry with "pending_request" set (MPI_Irecv) is also held back until
 *    "request_done" is set, because the number of received bytes is only known
 *    when the request has finished. The wait is measured from the time the
 *    receive was posted. If the receive is still pending after that time and
 *    it is a pending UCX receive (lifecycle state PENDING), it is abandoned:
 *    the slot is freed without writing the entry, and the completion of the
 *    receive writes a new entry when its message arrives. No data is lost and
 *    nothing waits behind an idle receive.
 *
 * Requires the type "qentry" (with the members "end" of type struct timespec,
 * "pending_request", "request_done" and "lifecycle") to be defined before this
 * header is included. It uses the GCC/Clang __atomic
 * builtins only, so that it can be tested without building Open MPI.
 */
#ifndef EDUMPI_RING_H
#define EDUMPI_RING_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#define EDUMPI_CPU_RELAX() __asm__ __volatile__("pause" ::: "memory")
#elif defined(__aarch64__)
#define EDUMPI_CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
#define EDUMPI_CPU_RELAX() __asm__ __volatile__("" ::: "memory")
#endif

#ifndef EDUMPI_LC
/* same definitions as in init.h (there is no init.h in the stand-alone test) */
enum {
    EDUMPI_LC_NONE = 0,
    EDUMPI_LC_PENDING = 1,
    EDUMPI_LC_FINISHING = 2,
    EDUMPI_LC_ABANDONED = 3
};
#define EDUMPI_LC(ticket, state) (((uint64_t)(ticket) << 2) | (uint64_t)(state))
static inline int edumpi_lifecycle_try_abandon(qentry *e)
{
    uint64_t lc = __atomic_load_n(&e->lifecycle, __ATOMIC_ACQUIRE);
    if ((lc & 3) != EDUMPI_LC_PENDING) {
        return 0;
    }
    return __atomic_compare_exchange_n(&e->lifecycle, &lc,
                                       (lc & ~(uint64_t) 3) | EDUMPI_LC_ABANDONED, 0,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
#endif

/*
 * Zero an entry for a new event and give it a new ticket. The lifecycle word is not zeroed
 * but stored atomically: the completion callback of an abandoned receive may still do a
 * compare-and-swap on it (which fails because the ticket is different now).
 */
static inline void edumpi_entry_reset(qentry *slot, uint64_t ticket)
{
    const size_t before = offsetof(qentry, lifecycle);
    const size_t after = before + sizeof(slot->lifecycle);
    memset(slot, 0, before);
    memset((char *) slot + after, 0, sizeof(*slot) - after);
    __atomic_store_n(&slot->lifecycle, EDUMPI_LC(ticket, EDUMPI_LC_NONE), __ATOMIC_RELEASE);
}

enum {
    EDUMPI_SLOT_FREE = 0,
    EDUMPI_SLOT_ACTIVE = 1 /* zeroed and handed to a producer */
};

typedef struct edumpi_amt_ring {
    qentry *slots;
    int32_t *states;
    int64_t size;
    int64_t stall_ns;        /* unfinished call (no end timestamp yet) */
    int64_t request_wait_ns; /* finished call, but its request is not */

    /* written by the producers */
    int64_t head __attribute__((aligned(64)));
    uint64_t dropped __attribute__((aligned(64)));

    /* written by the consumer only (read by the producers) */
    int64_t tail __attribute__((aligned(64)));

    /* only touched by the consumer */
    int stalled;
    struct timespec stall_since;
} edumpi_amt_ring_t;

typedef void (*edumpi_amt_emit_fn)(qentry *entry, void *ctx);

static inline int edumpi_amt_ring_init(edumpi_amt_ring_t *r, int64_t size,
                                       int64_t stall_ms, int64_t request_wait_ms)
{
    memset(r, 0, sizeof(*r));
    r->slots = (qentry *) calloc((size_t) size, sizeof(qentry));
    r->states = (int32_t *) calloc((size_t) size, sizeof(int32_t));
    if (NULL == r->slots || NULL == r->states) {
        free(r->slots);
        free(r->states);
        r->slots = NULL;
        r->states = NULL;
        return -1;
    }
    r->size = size;
    r->stall_ns = stall_ms * 1000000LL;
    r->request_wait_ns = request_wait_ms * 1000000LL;
    return 0;
}

static inline void edumpi_amt_ring_destroy(edumpi_amt_ring_t *r)
{
    free(r->slots);
    free(r->states);
    r->slots = NULL;
    r->states = NULL;
}

/* Claim a slot. Safe to call from any number of threads. Never blocks. */
static inline qentry *edumpi_amt_ring_claim(edumpi_amt_ring_t *r)
{
    /* Target of events that could not get a slot. Its content is never read.
     * It is per thread, so that dropping events never shares memory. */
    static __thread qentry scratch;
    int64_t h = __atomic_load_n(&r->head, __ATOMIC_RELAXED);

    for (;;) {
        int64_t t = __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE);
        if (h - t >= r->size) {
            /* Ring is full: do not block the application, drop the event. */
            __atomic_fetch_add(&r->dropped, 1, __ATOMIC_RELAXED);
            memset(&scratch, 0, sizeof(scratch));
            return &scratch;
        }
        /* On failure h is reloaded with the current head. */
        if (__atomic_compare_exchange_n(&r->head, &h, h + 1, 0,
                                        __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            int64_t idx = h % r->size;
            qentry *slot = &r->slots[idx];
            edumpi_entry_reset(slot, (uint64_t) h + 1); /* unique ticket */
            __atomic_store_n(&r->states[idx], EDUMPI_SLOT_ACTIVE,
                             __ATOMIC_RELEASE);
            return slot;
        }
    }
}

static inline int edumpi_amt_entry_complete(qentry *e)
{
    return 0 != __atomic_load_n(&e->end.tv_sec, __ATOMIC_ACQUIRE)
           || 0 != __atomic_load_n(&e->end.tv_nsec, __ATOMIC_ACQUIRE);
}

static inline int edumpi_amt_request_ready(qentry *e)
{
    return 0 == __atomic_load_n(&e->pending_request, __ATOMIC_RELAXED)
           || 0 != __atomic_load_n(&e->request_done, __ATOMIC_ACQUIRE);
}

/* age of an entry: the time since its call returned (end is a CLOCK_REALTIME stamp) */
static inline int64_t edumpi_amt_age_ns(qentry *e)
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    int64_t t_now = (int64_t) now.tv_sec * 1000000000LL + now.tv_nsec;
    int64_t t_end = (int64_t) __atomic_load_n(&e->end.tv_sec, __ATOMIC_ACQUIRE) * 1000000000LL
                    + __atomic_load_n(&e->end.tv_nsec, __ATOMIC_ACQUIRE);
    return t_now - t_end;
}

static inline int64_t edumpi_amt_monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/*
 * Consumer side (one thread only). Hands up to "max" entries to emit() in
 * claim order and returns how many it handed over. An incomplete entry at the
 * tail is waited for until the stall timeout expires. With force != 0
 * (shutdown) everything that is left is emitted, complete or not.
 */
static inline int64_t edumpi_amt_ring_drain(edumpi_amt_ring_t *r,
                                            edumpi_amt_emit_fn emit, void *ctx,
                                            int force, int64_t max)
{
    int64_t n = 0;

    while (n < max) {
        int64_t h = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
        int64_t tail = r->tail;
        if (tail >= h) {
            break;
        }
        int64_t idx = tail % r->size;
        int32_t state = __atomic_load_n(&r->states[idx], __ATOMIC_ACQUIRE);
        if (EDUMPI_SLOT_ACTIVE != state) {
            /* ticket taken, but the producer is still zeroing the slot */
            break;
        }

        qentry *e = &r->slots[idx];
        int ended = edumpi_amt_entry_complete(e);
        int ready = force || (ended && edumpi_amt_request_ready(e));
        if (!ready && ended) {
            /* The call returned, but its request has not finished. The wait is measured
             * from the time the receive was posted, so that a pool of idle receives is not
             * waited for one after the other. */
            if (edumpi_amt_age_ns(e) < r->request_wait_ns) {
                break;
            }
            if ((__atomic_load_n(&e->lifecycle, __ATOMIC_ACQUIRE) & 3) == EDUMPI_LC_FINISHING) {
                break; /* the completion is filling in the entry right now */
            }
            if (edumpi_lifecycle_try_abandon(e)) {
                /* Written when the message arrives: free the slot without writing it. */
                r->stalled = 0;
                __atomic_store_n(&r->states[idx], EDUMPI_SLOT_FREE, __ATOMIC_RELAXED);
                __atomic_store_n(&r->tail, tail + 1, __ATOMIC_RELEASE);
                n++;
                continue;
            }
            /* not a UCX receive: emit it as it is */
        } else if (!ready) {
            /* the call itself has not returned (a blocking call): wait for it, but not
             * longer than the stall timeout */
            int64_t now = edumpi_amt_monotonic_ns();
            if (!r->stalled) {
                r->stalled = 1;
                r->stall_since.tv_sec = now / 1000000000LL;
                r->stall_since.tv_nsec = now % 1000000000LL;
                break;
            }
            int64_t since = (int64_t) r->stall_since.tv_sec * 1000000000LL
                            + r->stall_since.tv_nsec;
            if (now - since < r->stall_ns) {
                break;
            }
            /* Not finished for too long: emit it as it is. */
        }

        emit(e, ctx);
        r->stalled = 0;
        __atomic_store_n(&r->states[idx], EDUMPI_SLOT_FREE, __ATOMIC_RELAXED);
        /* publishes the freed slot to the producers */
        __atomic_store_n(&r->tail, tail + 1, __ATOMIC_RELEASE);
        n++;
    }
    return n;
}

static inline uint64_t edumpi_amt_ring_dropped(edumpi_amt_ring_t *r)
{
    return __atomic_load_n(&r->dropped, __ATOMIC_RELAXED);
}

#endif /* EDUMPI_RING_H */
