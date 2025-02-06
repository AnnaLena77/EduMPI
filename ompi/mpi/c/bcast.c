/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2010-2012 Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/memchecker.h"
#include "ompi/runtime/ompi_spc.h"
#include <time.h>

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Bcast = PMPI_Bcast
#endif
#define MPI_Bcast PMPI_Bcast
#endif

static const char FUNC_NAME[] = "MPI_Bcast";


int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm)
{
#ifdef ENABLE_ANALYSIS
    qentry *item = getWritingRingPos();
    clock_gettime(CLOCK_REALTIME, &item->start);
    initQentry(&item, -1, "MPI_Bcast", 9, 0, 0, "collective", 10, datatype, NULL, comm, 1, NULL);
#endif 
    
    //Integer for the partnerrank-value= -1
    //-> Function Bcast sends the message to all other processes. All other processes are partnerranks
    int err;

    SPC_RECORD(OMPI_SPC_BCAST, 1);

    MEMCHECKER(
        memchecker_datatype(datatype);
        memchecker_comm(comm);
        if (OMPI_COMM_IS_INTRA(comm)) {
            if (ompi_comm_rank(comm) == root) {
                /* check whether root's send buffer is defined. */
                memchecker_call(&opal_memchecker_base_isdefined, buffer, count, datatype);
            } else {
                /* check whether receive buffer is addressable. */
                memchecker_call(&opal_memchecker_base_isaddressable, buffer, count, datatype);
            }
        } else {
            if (MPI_ROOT == root) {
                /* check whether root's send buffer is defined. */
                memchecker_call(&opal_memchecker_base_isdefined, buffer, count, datatype);
            } else if (MPI_PROC_NULL != root) {
                /* check whether receive buffer is addressable. */
                memchecker_call(&opal_memchecker_base_isaddressable, buffer, count, datatype);
            }
        }
    );

    if (MPI_PARAM_CHECK) {
      err = MPI_SUCCESS;
      OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
      if (ompi_comm_invalid(comm)) {
          return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_COMM,
                                     FUNC_NAME);
      }

      /* Errors for all ranks */

      OMPI_CHECK_DATATYPE_FOR_SEND(err, datatype, count);
      OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
      if (MPI_IN_PLACE == buffer) {
          return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, FUNC_NAME);
      }

      /* Errors for intracommunicators */

      if (OMPI_COMM_IS_INTRA(comm)) {
        if ((root >= ompi_comm_size(comm)) || (root < 0)) {
          return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
        }
      }

      /* Errors for intercommunicators */

      else {
        if (! ((root >= 0 && root < ompi_comm_remote_size(comm)) ||
               MPI_ROOT == root || MPI_PROC_NULL == root)) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
        }
      }
    } 

#if OPAL_ENABLE_FT_MPI
    /*
     * An early check, so as to return early if we are using a broken
     * communicator. This is not absolutely necessary since we will
     * check for this, and other, error conditions during the operation.
     */
    if( OPAL_UNLIKELY(!ompi_comm_iface_coll_check(comm, &err)) ) {
        OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
    }
#endif

    /* If there's only one node, or if the count is 0, we're done */

    if ((OMPI_COMM_IS_INTRA(comm) && ompi_comm_size(comm) <= 1) ||
        0 == count) {
        return MPI_SUCCESS;
    }

    /* Invoke the coll component to perform the back-end operation */
#ifndef ENABLE_ANALYSIS
    err = comm->c_coll->coll_bcast(buffer, count, datatype, root, comm,
                                  comm->c_coll->coll_bcast_module);
#else
    //printf("%s, %d\n", item->operation, strcmp(item->operation, "MPI_Bcast"));
    //printf("Test auf NULL 1: %d\n", item == NULL);
    err = comm->c_coll->coll_bcast(buffer, count, datatype, root, comm,
                                  comm->c_coll->coll_bcast_module, &item);
    //qentryIntoQueue(&item);
    clock_gettime(CLOCK_REALTIME, &item->end);
#endif

    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
