/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2014-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/runtime/opal.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/datatype/opal_datatype_internal.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define N 331

uint32_t remote_arch = 0xffffffff;
bool report_all_errors = true;

struct foo_t {
    int i[3];
    double d[3];
} foo = {0}, *bar = NULL;

struct pfoo_t {
    int i[2];
    double d[2];
} pfoo = {0}, *pbar = NULL;

static void print_hex(void* ptr, int count, char* epilog, char* prolog)
{
    if ( NULL != epilog) fprintf(stderr, "%s", epilog);
    for ( int i = 0; i < count; i++ ) {
        fprintf(stderr, "%02x", (unsigned int)(((unsigned char*)ptr)[i]));
    }
    if (NULL != prolog) fprintf(stderr, "%s", prolog);
}

static void print_bar_pbar(struct foo_t* _bar, struct pfoo_t* _pbar)
{
    print_hex(&_bar->i[0], sizeof(int), NULL, " ");
    print_hex(&_bar->i[1], sizeof(int), "[", "] ");
    print_hex(&_bar->i[2], sizeof(int), NULL, " ");
    print_hex(&_bar->d[0], sizeof(double), NULL, " ");
    print_hex(&_bar->d[1], sizeof(double), "[", "] ");
    print_hex(&_bar->d[2], sizeof(double), NULL, "\n");

    print_hex(&_pbar->i[0], sizeof(int), NULL, "            ");
    print_hex(&_pbar->i[1], sizeof(int), NULL, " ");
    print_hex(&_pbar->d[0], sizeof(double), NULL, "                    ");
    print_hex(&_pbar->d[1], sizeof(double), NULL, "\n");
}

static void print_stack(opal_convertor_t* conv)
{
    printf("Stack pos %d [converted %" PRIsize_t "/%" PRIsize_t "]\n",
           conv->stack_pos, conv->bConverted, conv->local_size);
    for( uint32_t i = 0; i <= conv->stack_pos; i++ ) {
        printf( "[%u] index %d, type %s count %" PRIsize_t " disp %p\n",
                i, conv->pStack[i].index, opal_datatype_basicDatatypes[conv->pStack[i].type]->name,
                conv->pStack[i].count, (void*)conv->pStack[i].disp);
    }
    printf("\n");
}

static int testcase(ompi_datatype_t * newtype, size_t arr[][2]) {
    int i, j, errors = 0;
    struct iovec a;
    unsigned int iov_count;
    size_t max_data;
    size_t pos;
    opal_convertor_t * pConv;

    for (j = 0; j < N; ++j) {
        pbar[j].i[0] = 123+j;
        pbar[j].i[1] = 789+j;
        pbar[j].d[0] = 123.456+j;
        pbar[j].d[1] = 789.123+j;
        memset(&bar[j].i[0], 0xFF, sizeof(int));
        memset(&bar[j].i[2], 0xFF, sizeof(int));
        bar[j].i[1] = 0;
        memset(&bar[j].d[0], 0xFF, sizeof(double));
        memset(&bar[j].d[2], 0xFF, sizeof(double));
        bar[j].d[1] = 0.0;
    }

    pConv = opal_convertor_create( remote_arch, 0 );
    if( OPAL_SUCCESS != opal_convertor_prepare_for_recv( pConv, &(newtype->super), N, bar ) ) {
        printf( "Cannot attach the datatype to a convertor\n" );
        return OMPI_ERROR;
    }

    for ( i = 0; 0 != arr[i][0]; i++) {
        /* add some garbage before and after the source data */
        a.iov_base = malloc(arr[i][0]+2048);
        if (NULL == a.iov_base) {
            printf("cannot malloc iov_base\n");
            return 1;
        }
        memset(a.iov_base, 0xAA, 1024);
        memcpy((char*)a.iov_base+1024, (char *)pbar + arr[i][1], arr[i][0]);
        memset((char*)a.iov_base+1024+arr[i][0], 0xAA, 1024);
        a.iov_base = (char*)a.iov_base + 1024;
        a.iov_len = arr[i][0];
        iov_count = 1;
        max_data = a.iov_len;
        pos = arr[i][1];
        opal_convertor_set_position(pConv, &pos);
        print_stack(pConv);
        assert(arr[i][1] == pos);
        opal_convertor_unpack( pConv, &a, &iov_count, &max_data );
        a.iov_base = (char*)a.iov_base - 1024;
        free(a.iov_base);
    }

    for (j = 0; j < N; ++j) {
        if (bar[j].i[0] != pbar[j].i[0] ||
            bar[j].i[1] != 0 ||
            bar[j].i[2] != pbar[j].i[1] ||
            bar[j].d[0] != pbar[j].d[0] ||
            bar[j].d[1] != 0.0 ||
            bar[j].d[2] != pbar[j].d[1]) {
            if(0 == errors || report_all_errors) {
                ptrdiff_t displ;
                char* error_location = "in gaps";
                if (bar[j].i[0] != pbar[j].i[0]) {
                    displ = (char*)&bar[j].i[0] - (char*)&bar[0];
                    error_location = "i[0]";
                } else if (bar[j].i[2] != pbar[j].i[1]) {
                    displ = (char*)&bar[j].i[1] - (char*)&bar[0];
                    error_location = "i[2]";
                } else if (bar[j].d[0] != pbar[j].d[0]) {
                    displ = (char*)&bar[j].d[0] - (char*)&bar[0];
                    error_location = "d[0]";
                } else if (bar[j].d[2] != pbar[j].d[1]) {
                    displ = (char*)&bar[j].d[1] - (char*)&bar[0];
                    error_location = "d[2]";
                } else {
                    displ = (char*)&bar[j] - (char*)&bar[0];
                }
                for (i = 0; 0 != arr[i][0]; i++) {
                    if( (displ >= arr[i][1]) && (displ <= (arr[i][1] + arr[i][0])) ) {
                        fprintf(stderr, "Problem encountered %li bytes into the %d unpack [%"PRIsize_t":%"PRIsize_t"]\n",
                                displ - arr[i][1], i, arr[i][1], arr[i][0]);
                        break;
                    }
                }

                (void)opal_datatype_dump(&newtype->super);
                fprintf(stderr, "ERROR ! struct %d/%d in field %s, ptr = %p"
                        " got (%d,%d,%d,%g,%g,%g) expected (%d,%d,%d,%g,%g,%g)\n",
                        j, N, error_location, (void*)&bar[j],
                        bar[j].i[0],
                        bar[j].i[1],
                        bar[j].i[2],
                        bar[j].d[0],
                        bar[j].d[1],
                        bar[j].d[2],
                        pbar[j].i[0],
                        0,
                        pbar[j].i[1],
                        pbar[j].d[0],
                        0.0,
                        pbar[j].d[1]);
                print_bar_pbar(&bar[j], &pbar[j]);
                if( report_all_errors ) fprintf(stderr, "\n\n");
            }
            errors++;
        }
    }
    OBJ_RELEASE( pConv );
    return errors;
}

static int unpack_ooo(void)
{
    ompi_datatype_t * t1;
    ompi_datatype_t * t2;
    ompi_datatype_t * type[2];
    ompi_datatype_t * newtype;
    MPI_Aint disp[2];
    int len[2], rc;

    rc = ompi_datatype_create_vector(2, 1, 2, MPI_INT, &t1);
    if (OMPI_SUCCESS != rc) {
        fprintf(stderr, "could not create vector t1\n");
        return 1;
    }
    rc = ompi_datatype_commit (&t1);
    if (OMPI_SUCCESS != rc) {
        fprintf(stderr, "could not commit vector t1\n");
        return 1;
    }

    rc = ompi_datatype_create_vector(2, 1, 2, MPI_DOUBLE, &t2);
    if (OMPI_SUCCESS != rc) {
        fprintf(stderr, "could not create vector t2\n");
        return 1;
    }
    rc = ompi_datatype_commit (&t2);
    if (OMPI_SUCCESS != rc) {
        fprintf(stderr, "could not commit vector t2\n");
        return 1;
    }

/*
 * btl=0x7f7823672580 bytes_received=992 data_offset=0
 * btl=0x7f7823260420 bytes_received=1325 data_offset=992
 * btl=0x7f7823672580 bytes_received=992 data_offset=2317
 * btl=0x7f7823672580 bytes_received=992 data_offset=3309
 * btl=0x7f7823672580 bytes_received=992 data_offset=4301
 * btl=0x7f7823672580 bytes_received=992 data_offset=5293
 * btl=0x7f7823672580 bytes_received=992 data_offset=6285
 * btl=0x7f7823672580 bytes_received=667 data_offset=7277
 */
    size_t test1[9][2] = {
        {992, 0},
        {1325, 0 + 992},
        {992, 992 + 1325 /* = 2317 */},
        {992, 2317 + 992 /* = 3309 */},
        {992, 3309 + 992 /* = 4301 */},
        {992, 4301 + 992 /* = 5293 */},
        {992, 5293 + 992 /* = 6285 */},
        {667, 6285 + 992 /* = 7277 */},
        {0, -1},
    };

/*
 * btl=0x7f80bc545580 bytes_received=992 data_offset=0
 * btl=0x7f80bc545580 bytes_received=992 data_offset=2317
 * btl=0x7f80bc545580 bytes_received=992 data_offset=3309
 * btl=0x7f80bc545580 bytes_received=992 data_offset=4301
 * btl=0x7f80bc545580 bytes_received=992 data_offset=5293
 * btl=0x7f80bc545580 bytes_received=992 data_offset=6285
 * btl=0x7f80bc133420 bytes_received=1325 data_offset=992
 * btl=0x7f80bc545580 bytes_received=667 data_offset=7277
 */
    size_t test2[9][2] = {
        {992, 0},
        {992, 2317},
        {992, 3309},
        {992, 4301},
        {992, 5293},
        {992, 6285},
        {1325, 992},
        {667, 7277},
        {0, -1},
    };

/* trimmed version of test2 */
    size_t test3[9][2] = {
        {992, 0},
        {4960, 2317},
        {1325, 992},
        {667, 7277},
        {0, -1},
    };

/* an other test case */
    size_t test4[9][2] = {
        {992, 0},
        {992, 2976},
        {992, 1984},
        {992, 992},
        {3976, 3968},
        {0, -1},
    };

    disp[0] = (long)(&foo.i[0]) - (long)&foo;
    disp[1] = (long)(&foo.d[0]) - (long)&foo;

    type[0] = t1;
    type[1] = t2;

    len[0] = 1;
    len[1] = 1;

    rc = ompi_datatype_create_struct(2, len, disp, type, &newtype);
    if (OMPI_SUCCESS != rc) {
        fprintf(stderr, "could not create struct\n");
        return 1;
    }
    rc = ompi_datatype_commit (&newtype);
    if (OMPI_SUCCESS != rc) {
        fprintf(stderr, "could not create struct\n");
        return 1;
    }

    pbar = (struct pfoo_t *)malloc (N * sizeof(struct pfoo_t));
    if (NULL == pbar) {
        fprintf(stderr, "could not malloc pbar\n");
        return 1;
    }
    bar = (struct foo_t *)malloc (N * sizeof(struct foo_t));
    if (NULL == bar) {
        fprintf(stderr, "could not malloc bar\n");
        return 1;
    }

    if (0 != testcase(newtype, test1)) {
        printf ("test1 failed\n");
        return 2;
    }

    if (0 != testcase(newtype, test2)) {
        printf ("test2 failed\n");
        return 2;
    }

    if (0 != testcase(newtype, test3)) {
        printf ("test3 failed\n");
        return 2;
    }

    if (0 != testcase(newtype, test4)) {
        printf ("test4 failed\n");
        return 2;
    }


    /* test the automatic destruction pf the data */
    ompi_datatype_destroy( &newtype ); assert( newtype == NULL );

    return rc;
}

int main( int argc, char* argv[] )
{
    int rc;

    opal_init(&argc, &argv);
    ompi_datatype_init();
    /**
     * By default simulate homogeneous architectures.
     */
    remote_arch = opal_local_arch;

    printf( "\n\n#\n * TEST UNPACK OUT OF ORDER\n #\n\n" );
    rc = unpack_ooo();
    if( rc == 0 ) {
        printf( "unpack out of order [PASSED]\n" );
        return 0;
    }
    printf( "unpack out of order [NOT PASSED]\n" );
    return -1;
}
