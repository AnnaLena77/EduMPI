/*
 * Stand-alone test of the multi-producer ring (ompi/mpi/c/edumpi_ring.h).
 * It does not need a built Open MPI or a database:
 *
 *   cc -O2 -pthread -I../.. amt_ring_test.c -o amt_ring_test && ./amt_ring_test
 *   cc -O1 -g -fsanitize=thread -pthread -I../.. amt_ring_test.c -o amt_ring_test_tsan && ./amt_ring_test_tsan
 *
 * (run from this directory; -I../.. is the top of the source tree)
 */
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/* Minimal stand-in for the real qentry, with the members the ring needs. */
typedef struct qentry {
    int producer;
    int64_t seq;
    int64_t check;
    char pad[800];
    int64_t recv_size;
    int pending_request;
    int request_done;
    struct timespec end;
} qentry;

#include "ompi/mpi/c/edumpi_ring.h"

static int failures = 0;
#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                      \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static void complete(qentry *e)
{
    /* what the MPI wrappers do when the call returns: set "end" last */
    __atomic_store_n(&e->end.tv_nsec, 1 + (long) (e->seq % 1000), __ATOMIC_RELEASE);
    __atomic_store_n(&e->end.tv_sec, 1, __ATOMIC_RELEASE);
}

/* ------------------------------------------------------------------ test 1 */
#define NPROD 8
#define NEVENTS 200000

typedef struct {
    int64_t last_seq[NPROD];
    int64_t count[NPROD];
    int64_t total;
    int bad_order, bad_check, incomplete;
} seen_t;

static void count_emit(qentry *e, void *ctx)
{
    seen_t *s = (seen_t *) ctx;
    int p = e->producer;
    if (p < 0 || p >= NPROD) {
        s->bad_check++;
        return;
    }
    if (e->check != (int64_t) p * 1000000000LL + e->seq) {
        s->bad_check++;
    }
    if (e->seq <= s->last_seq[p]) {
        s->bad_order++;
    }
    if (!edumpi_amt_entry_complete(e)) {
        s->incomplete++;
    }
    s->last_seq[p] = e->seq;
    s->count[p]++;
    s->total++;
}

typedef struct {
    edumpi_amt_ring_t *ring;
    int id;
} prod_arg_t;

static void *producer(void *arg)
{
    prod_arg_t *a = (prod_arg_t *) arg;
    for (int64_t i = 1; i <= NEVENTS; i++) {
        qentry *e = edumpi_amt_ring_claim(a->ring);
        e->producer = a->id;
        e->seq = i;
        e->check = (int64_t) a->id * 1000000000LL + i;
        complete(e);
    }
    return NULL;
}

typedef struct {
    edumpi_amt_ring_t *ring;
    seen_t *seen;
    int stop;
} cons_arg_t;

static void *consumer(void *arg)
{
    cons_arg_t *c = (cons_arg_t *) arg;
    while (!__atomic_load_n(&c->stop, __ATOMIC_ACQUIRE)) {
        if (0 == edumpi_amt_ring_drain(c->ring, count_emit, c->seen, 0, 4096)) {
            struct timespec ts = {0, 100000};
            nanosleep(&ts, NULL);
        }
    }
    while (edumpi_amt_ring_drain(c->ring, count_emit, c->seen, 1, 4096) > 0) {
    }
    return NULL;
}

static void test_many_producers(int64_t ring_size)
{
    edumpi_amt_ring_t ring;
    seen_t seen;
    memset(&seen, 0, sizeof(seen));
    for (int p = 0; p < NPROD; p++) {
        seen.last_seq[p] = 0;
    }
    CHECK(0 == edumpi_amt_ring_init(&ring, ring_size, 10000, 2000), "init");

    cons_arg_t carg = {&ring, &seen, 0};
    pthread_t cons, prods[NPROD];
    prod_arg_t pargs[NPROD];
    pthread_create(&cons, NULL, consumer, &carg);
    for (int p = 0; p < NPROD; p++) {
        pargs[p].ring = &ring;
        pargs[p].id = p;
        pthread_create(&prods[p], NULL, producer, &pargs[p]);
    }
    for (int p = 0; p < NPROD; p++) {
        pthread_join(prods[p], NULL);
    }
    __atomic_store_n(&carg.stop, 1, __ATOMIC_RELEASE);
    pthread_join(cons, NULL);

    uint64_t dropped = edumpi_amt_ring_dropped(&ring);
    printf("ring %" PRId64 ": emitted %" PRId64 ", dropped %" PRIu64
           ", total %d\n",
           ring_size, seen.total, dropped, NPROD * NEVENTS);
    CHECK(seen.total + (int64_t) dropped == (int64_t) NPROD * NEVENTS,
          "every event is either emitted or counted as dropped");
    CHECK(0 == seen.bad_order, "per-producer order violated %d times", seen.bad_order);
    CHECK(0 == seen.bad_check, "corrupted entries: %d", seen.bad_check);
    CHECK(0 == seen.incomplete, "incomplete entries emitted: %d", seen.incomplete);
    edumpi_amt_ring_destroy(&ring);
}

/* ------------------------------------------------------------------ test 2 */
static void noop_emit(qentry *e, void *ctx)
{
    (void) e;
    (*(int64_t *) ctx)++;
}

static void test_ring_full_drops(void)
{
    edumpi_amt_ring_t ring;
    int64_t emitted = 0;
    CHECK(0 == edumpi_amt_ring_init(&ring, 1024, 10000, 2000), "init");
    for (int i = 0; i < 5000; i++) {
        qentry *e = edumpi_amt_ring_claim(&ring);
        e->seq = i + 1;
        complete(e);
    }
    CHECK(3976 == (int64_t) edumpi_amt_ring_dropped(&ring),
          "dropped %" PRIu64 ", expected 3976", edumpi_amt_ring_dropped(&ring));
    while (edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100) > 0) {
    }
    CHECK(1024 == emitted, "emitted %" PRId64 ", expected 1024", emitted);
    /* the ring is usable again after it was drained */
    qentry *e = edumpi_amt_ring_claim(&ring);
    complete(e);
    CHECK(3976 == (int64_t) edumpi_amt_ring_dropped(&ring), "no new drop after drain");
    edumpi_amt_ring_destroy(&ring);
}

/* ------------------------------------------------------------------ test 3 */
static void test_stall_timeout(void)
{
    edumpi_amt_ring_t ring;
    int64_t emitted = 0;
    CHECK(0 == edumpi_amt_ring_init(&ring, 64, 200, 2000), "init"); /* 200 ms */

    qentry *stuck = edumpi_amt_ring_claim(&ring); /* never completed */
    stuck->seq = 1;
    for (int i = 0; i < 5; i++) {
        qentry *e = edumpi_amt_ring_claim(&ring);
        e->seq = 2 + i;
        complete(e);
    }
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(0 == emitted, "nothing may pass the incomplete entry yet (%" PRId64 ")", emitted);
    usleep(300 * 1000);
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(6 == emitted, "after the timeout all 6 are emitted (%" PRId64 ")", emitted);
    edumpi_amt_ring_destroy(&ring);
}

/* ------------------------------------------------------------------ test 4 */
static void test_force_drain(void)
{
    edumpi_amt_ring_t ring;
    int64_t emitted = 0;
    CHECK(0 == edumpi_amt_ring_init(&ring, 64, 60000, 2000), "init");
    for (int i = 0; i < 10; i++) {
        qentry *e = edumpi_amt_ring_claim(&ring);
        e->seq = i + 1; /* none of them completed */
    }
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(0 == emitted, "not complete, no force: %" PRId64, emitted);
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 1, 100);
    CHECK(10 == emitted, "force drains everything: %" PRId64, emitted);
    edumpi_amt_ring_destroy(&ring);
}

/* ------------------------------------------------------------------ test 5 */
/* An MPI_Irecv entry waits for its request; a timeout releases it. */
static void test_pending_request(void)
{
    edumpi_amt_ring_t ring;
    int64_t emitted = 0;
    CHECK(0 == edumpi_amt_ring_init(&ring, 64, 10000, 200), "init"); /* request wait 200 ms */

    qentry *irecv = edumpi_amt_ring_claim(&ring);
    irecv->seq = 1;
    irecv->pending_request = 1; /* set by the wrapper before the PML call */
    complete(irecv);            /* the wrapper returned */
    for (int i = 0; i < 3; i++) {
        qentry *e = edumpi_amt_ring_claim(&ring);
        e->seq = 2 + i;
        complete(e);
    }
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(0 == emitted, "entries behind a pending request wait (%" PRId64 ")", emitted);

    __atomic_store_n(&irecv->recv_size, 4, __ATOMIC_RELAXED);
    __atomic_store_n(&irecv->request_done, 1, __ATOMIC_RELEASE); /* the PML finished it */
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(4 == emitted, "all 4 are emitted once the request is done (%" PRId64 ")", emitted);

    /* a request that never finishes is released by the timeout */
    qentry *idle = edumpi_amt_ring_claim(&ring);
    idle->pending_request = 1;
    complete(idle);
    qentry *behind = edumpi_amt_ring_claim(&ring);
    complete(behind);
    emitted = 0;
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(0 == emitted, "idle request holds back (%" PRId64 ")", emitted);
    usleep(300 * 1000);
    edumpi_amt_ring_drain(&ring, noop_emit, &emitted, 0, 100);
    CHECK(2 == emitted, "timeout releases it (%" PRId64 ")", emitted);
    edumpi_amt_ring_destroy(&ring);
}

/* A completion thread sets the size later, like the UCX completion callback. */
#define NREQ 20000
typedef struct {
    qentry *slots[NREQ];
    int next_to_finish;
    int stop;
} completer_t;

static void *completer(void *arg)
{
    completer_t *c = (completer_t *) arg;
    int done = 0;
    while (done < NREQ) {
        int published = __atomic_load_n(&c->next_to_finish, __ATOMIC_ACQUIRE);
        while (done < published) {
            qentry *e = c->slots[done++];
            __atomic_store_n(&e->recv_size, 4, __ATOMIC_RELAXED);
            __atomic_store_n(&e->request_done, 1, __ATOMIC_RELEASE);
        }
        struct timespec ts = {0, 20000};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

typedef struct {
    int64_t total, without_size;
} size_seen_t;

static void size_emit(qentry *e, void *ctx)
{
    size_seen_t *s = (size_seen_t *) ctx;
    s->total++;
    if (4 != __atomic_load_n(&e->recv_size, __ATOMIC_RELAXED)) {
        s->without_size++;
    }
}

static void test_async_completion(void)
{
    static completer_t comp;
    edumpi_amt_ring_t ring;
    size_seen_t seen = {0, 0};
    CHECK(0 == edumpi_amt_ring_init(&ring, 1 << 16, 10000, 5000), "init");
    pthread_t th;
    pthread_create(&th, NULL, completer, &comp);
    for (int i = 0; i < NREQ; i++) {
        qentry *e = edumpi_amt_ring_claim(&ring);
        e->pending_request = 1;
        complete(e);
        comp.slots[i] = e;
        __atomic_store_n(&comp.next_to_finish, i + 1, __ATOMIC_RELEASE);
        edumpi_amt_ring_drain(&ring, size_emit, &seen, 0, 4096);
    }
    pthread_join(th, NULL);
    while (edumpi_amt_ring_drain(&ring, size_emit, &seen, 0, 4096) > 0) {
    }
    CHECK(NREQ == seen.total, "all %d entries emitted (%" PRId64 ")", NREQ, seen.total);
    CHECK(0 == seen.without_size, "%" PRId64 " entries were written before their size was known",
          seen.without_size);
    edumpi_amt_ring_destroy(&ring);
}

int main(void)
{
    test_pending_request();
    test_async_completion();
    test_ring_full_drops();
    test_stall_timeout();
    test_force_drain();
    test_many_producers(1 << 20); /* big ring: nothing should be dropped */
    test_many_producers(4096);    /* small ring: consumer may fall behind */
    if (0 == failures) {
        printf("ALL TESTS PASSED\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
