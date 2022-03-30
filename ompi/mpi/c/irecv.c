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
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2012 Oak Ridge National Labs.  All rights reserved.
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
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/request/request.h"
#include "ompi/memchecker.h"
#include "ompi/runtime/ompi_spc.h"
#include <time.h>

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Irecv = PMPI_Irecv
#endif
#define MPI_Irecv PMPI_Irecv
#endif

static const char FUNC_NAME[] = "MPI_Irecv";


int MPI_Irecv(void *buf, int count, MPI_Datatype type, int source,
              int tag, MPI_Comm comm, MPI_Request *request)
{
    #ifdef ENABLE_ANALYSIS
    qentry *item = (qentry*)malloc(sizeof(qentry));
    initQentry(&item);
    //item->start
    time_t current_time = time(NULL);
    item->start = current_time;
    //item->operation
    strcpy(item->operation, "MPI_Irecv");
    //item->blocking
    item->blocking = 0;
    //item->datatype
    char *type_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int type_name_length;
    MPI_Type_get_name(type, type_name, &type_name_length);
    strcpy(item->datatype, type_name);
    free(type_name);
    //item->count
    item->count = count;
    //item->datasize
    item->datasize = count * sizeof(type);
    //item->communicator
    char *comm_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int comm_name_length;
    MPI_Comm_get_name(comm, comm_name, &comm_name_length);
    strcpy(item->communicator, comm_name);
    free(comm_name);
    //item->processrank
    int processrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processrank);
    item->processrank = processrank;
    //item->partnerrank
    item->partnerrank = source;
    #endif

    int rc = MPI_SUCCESS;

    SPC_RECORD(OMPI_SPC_IRECV, 1);

    MEMCHECKER(
        memchecker_datatype(type);
        memchecker_comm(comm);
    );

    if ( MPI_PARAM_CHECK ) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        OMPI_CHECK_DATATYPE_FOR_RECV(rc, type, count);
        OMPI_CHECK_USER_BUFFER(rc, buf, type, count);

        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_COMM, FUNC_NAME);
        } else if (((tag < 0) && (tag != MPI_ANY_TAG)) || (tag > mca_pml.pml_max_tag)) {
            rc = MPI_ERR_TAG;
        } else if ((MPI_ANY_SOURCE != source) &&
                   (MPI_PROC_NULL != source) &&
                   ompi_comm_peer_invalid(comm, source)) {
            rc = MPI_ERR_RANK;
        } else if (NULL == request) {
            rc = MPI_ERR_REQUEST;
        }
        OMPI_ERRHANDLER_CHECK(rc, comm, rc, FUNC_NAME);
    }

    if (source == MPI_PROC_NULL) {
        *request = &ompi_request_empty;
        return MPI_SUCCESS;
    }

#if OPAL_ENABLE_FT_MPI
    /*
     * The request will be checked for process failure errors during the
     * completion calls. So no need to check here.
     */
#endif

    MEMCHECKER (
        memchecker_call(&opal_memchecker_base_mem_noaccess, buf, count, type);
    );
#ifndef ENABLE_ANALYSIS
    rc = MCA_PML_CALL(irecv(buf,count,type,source,tag,comm,request));
#else
    rc = MCA_PML_CALL(irecv(buf,count,type,source,tag,comm,request, &item));
#endif
    OMPI_ERRHANDLER_RETURN(rc, comm, rc, FUNC_NAME);
}
