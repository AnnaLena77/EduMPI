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
    uint64_t lifecycle;
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
    /* what the MPI wrappers do when the call returns: stamp "end" with the real time (the
     * request wait is measured from it), the seconds last */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    __atomic_store_n(&e->end.tv_nsec, ts.tv_nsec + 1, __ATOMIC_RELEASE);
    __atomic_store_n(&e->end.tv_sec, ts.tv_sec, __ATOMIC_RELEASE);
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

/* ------------------------------------------------------------------ test 6 */
/*
 * Abandoning a pending receive and completing it late. Like the UCX PML does it: the
 * monitor thread abandons receives that wait longer than the limit, the completion
 * callback finishes the entry in place if it still can and otherwise writes a new entry.
 * The ring is small and the completions come around the limit, so the slots of abandoned
 * entries are reused all the time. Every request must end up as exactly one entry with data.
 */
#ifndef NABAND
#define NABAND 30000
#endif

typedef struct {
    edumpi_amt_ring_t *ring;
    qentry *slots[NABAND];
    uint64_t ticket[NABAND];
    int64_t post_ns[NABAND];
    int published;
    int completed;
} aband_t;

static void finish_request(aband_t *a, int i)
{
    qentry *q = a->slots[i];
    uint64_t expected = EDUMPI_LC(a->ticket[i], EDUMPI_LC_PENDING);
    if (__atomic_compare_exchange_n(&q->lifecycle, &expected,
                                    EDUMPI_LC(a->ticket[i], EDUMPI_LC_FINISHING), 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        /* in place */
        __atomic_store_n(&q->recv_size, 4, __ATOMIC_RELAXED);
        __atomic_store_n(&q->request_done, 1, __ATOMIC_RELEASE);
    } else {
        /* abandoned: a new entry with the data */
        qentry *n = edumpi_amt_ring_claim(a->ring);
        n->seq = i + 1;
        n->recv_size = 4;
        complete(n);
    }
}

static void *aband_completer(void *arg)
{
    aband_t *a = (aband_t *) arg;
    unsigned seed = 12345;
    for (int i = 0; i < NABAND;) {
        int published = __atomic_load_n(&a->published, __ATOMIC_ACQUIRE);
        if (i >= published) {
            struct timespec ts = {0, 50000};
            nanosleep(&ts, NULL);
            continue;
        }
        /* the completion arrives 0 to 6 ms after the request was posted, around the limit
         * of 3 ms: some before, some after it */
        int64_t due = a->post_ns[i] + (int64_t) (rand_r(&seed) % 6000) * 1000;
        while (edumpi_amt_monotonic_ns() < due) {
            struct timespec ts = {0, 20000};
            nanosleep(&ts, NULL);
        }
        finish_request(a, i++);
        __atomic_store_n(&a->completed, i, __ATOMIC_RELEASE);
    }
    return NULL;
}

typedef struct {
    int with_data[NABAND + 1];
    int without_data;
    int bad;
} aband_seen_t;

static void aband_emit(qentry *e, void *ctx)
{
    aband_seen_t *s = (aband_seen_t *) ctx;
    if (4 != __atomic_load_n(&e->recv_size, __ATOMIC_RELAXED)) {
        s->without_data++;
        return;
    }
    if (e->seq < 1 || e->seq > NABAND) {
        s->bad++;
        return;
    }
    s->with_data[e->seq]++;
}

static void test_abandon_race(void)
{
    static aband_t a;
    static aband_seen_t seen;
    edumpi_amt_ring_t ring;
    CHECK(0 == edumpi_amt_ring_init(&ring, 1024, 10000, 3), "init"); /* request wait 3 ms */
    a.ring = &ring;
    pthread_t th;
    pthread_create(&th, NULL, aband_completer, &a);

    for (int i = 0; i < NABAND; i++) {
        qentry *e = edumpi_amt_ring_claim(&ring);
        e->seq = i + 1;
        e->pending_request = 1;
        a.ticket[i] = __atomic_load_n(&e->lifecycle, __ATOMIC_RELAXED) >> 2;
        __atomic_store_n(&e->lifecycle, EDUMPI_LC(a.ticket[i], EDUMPI_LC_PENDING), /* as the UCX attach does */
                         __ATOMIC_RELEASE);
        a.slots[i] = e;
        a.post_ns[i] = edumpi_amt_monotonic_ns();
        complete(e);
        __atomic_store_n(&a.published, i + 1, __ATOMIC_RELEASE);
        edumpi_amt_ring_drain(&ring, aband_emit, &seen, 0, 4096);
        /* a realistic rate: about 20 microseconds per request, so fewer new requests within
         * the wait of 3 ms than the ring has slots (a busy wait: nanosleep is too coarse) */
        int64_t until = edumpi_amt_monotonic_ns() + 20000;
        while (edumpi_amt_monotonic_ns() < until) {
        }
    }
    /* keep draining while the completions are still coming in (the late ones need slots) */
    while (__atomic_load_n(&a.completed, __ATOMIC_ACQUIRE) < NABAND) {
        edumpi_amt_ring_drain(&ring, aband_emit, &seen, 0, 4096);
        struct timespec ts = {0, 100000};
        nanosleep(&ts, NULL);
    }
    pthread_join(th, NULL);
    /* let the remaining ones age and be abandoned or finished, then drain what is left */
    for (int round = 0; round < 50; round++) {
        edumpi_amt_ring_drain(&ring, aband_emit, &seen, 0, 4096);
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, NULL);
    }
    while (edumpi_amt_ring_drain(&ring, aband_emit, &seen, 1, 4096) > 0) {
    }

    int exactly_once = 0, missing = 0, twice = 0;
    for (int i = 1; i <= NABAND; i++) {
        if (1 == seen.with_data[i]) exactly_once++;
        else if (0 == seen.with_data[i]) missing++;
        else twice++;
    }
    printf("abandon race: %d of %d requests written exactly once with data, %d missing, "
           "%d twice, %d entries without data, dropped %" PRIu64 "\n",
           exactly_once, NABAND, missing, twice, seen.without_data,
           edumpi_amt_ring_dropped(&ring));
    CHECK(0 == seen.bad, "entries with a wrong id: %d", seen.bad);
    CHECK(0 == seen.without_data, "no entry may be written without its data (%d)", seen.without_data);
    CHECK(0 == twice, "no request may be written twice (%d)", twice);
    CHECK(0 == missing, "requests lost: %d", missing);
    CHECK(0 == edumpi_amt_ring_dropped(&ring), "events dropped: %" PRIu64,
          edumpi_amt_ring_dropped(&ring));
    edumpi_amt_ring_destroy(&ring);
}

int main(int argc, char **argv)
{
    /* "./amt_ring_test abandon" runs only the abandon-race test */
    if (argc > 1 && 0 == strcmp(argv[1], "abandon")) {
        test_abandon_race();
        printf(0 == failures ? "ALL TESTS PASSED\n" : "%d FAILURE(S)\n", failures);
        return failures ? 1 : 0;
    }
    test_abandon_race();
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
