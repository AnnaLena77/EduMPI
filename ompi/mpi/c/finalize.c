/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/mpi/c/init.h"
#include <pthread.h>

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Finalize = PMPI_Finalize
#endif
#define MPI_Finalize PMPI_Finalize
#endif

static const char FUNC_NAME[] = "MPI_Finalize";


int MPI_Finalize(void)
{
    /* If --with-spc and ompi_mpi_spc_dump_enabled were specified, print
     * all of the final SPC values aggregated across the whole MPI run.
     * Also, free all SPC memory.
     */
    SPC_FINI();


    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
    }

    /* Pretty simple */

    //sleep(8);
#ifdef ENABLE_ANALYSIS
    struct timespec start, end;
    long seconds, nanoseconds;
    double elapsed;
    //clock_gettime(CLOCK_MONOTONIC, &start);
    //printf("run_thread wird 0\n");
    //printf("Reader: %d, Writer: %d\n", reader_pos, writer_pos);
    run_thread = 0;
    //closeFile();
    pthread_join(MONITOR_THREAD, NULL);
    //clock_gettime(CLOCK_MONOTONIC, &end);
    //free(ringbuffer);
    
    /*seconds = end.tv_sec - start.tv_sec;
    nanoseconds = end.tv_nsec - start.tv_nsec;
    elapsed = seconds + nanoseconds*1e-9;

    printf("pthread_join dauerte %f Sekunden\n", elapsed);*/
    //closeFile();
#endif
    return ompi_mpi_finalize();
}
