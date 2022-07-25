/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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
 * Copyright (c) 2007      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2012 Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2015 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015      Mellanox Technologies. All rights reserved.
 *
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

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Alltoall = PMPI_Alltoall
#endif
#define MPI_Alltoall PMPI_Alltoall
#endif

static const char FUNC_NAME[] = "MPI_Alltoall";

//sendet Daten von allen beteiligten Prozessen eines gegebenen Kommunikators an alle diese Prozesse
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 MPI_Comm comm)
{
#ifdef ENABLE_ANALYSIS
    qentry *item = (qentry*)malloc(sizeof(qentry));
    initQentry(&item);
    gettimeofday(&item->start, NULL);
    strcpy(item->function, "MPI_Alltoall");
    strcpy(item->communicationType, "collective");
    //item->datatype
    char *type_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int type_name_length;
    MPI_Type_get_name(sendtype, type_name, &type_name_length);
    strcpy(item->datatype, type_name);
    free(type_name);

    //item->communicator
    char *comm_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int comm_name_length;
    MPI_Comm_get_name(comm, comm_name, &comm_name_length);
    strcpy(item->communicationArea, comm_name);
    free(comm_name);
    //item->processrank
    int processrank;
    MPI_Comm_rank(comm, &processrank);
    item->processrank = processrank;
    //item->partnerrank
    item->partnerrank = -1;


    item->blocking = 1;
#endif 
    int err;
    size_t recvtype_size;

    SPC_RECORD(OMPI_SPC_ALLTOALL, 1);

    MEMCHECKER(
        memchecker_comm(comm);
        if (MPI_IN_PLACE != sendbuf) {
            memchecker_datatype(sendtype);
            memchecker_call(&opal_memchecker_base_isdefined, (void *)sendbuf, sendcount, sendtype);
        }
        memchecker_datatype(recvtype);
        memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, recvcount, recvtype);
    );

    if (MPI_PARAM_CHECK) {

        /* Unrooted operation -- same checks for all ranks on both
           intracommunicators and intercommunicators */

        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_COMM,
                                          FUNC_NAME);
        } else if ((MPI_IN_PLACE == sendbuf && OMPI_COMM_IS_INTER(comm)) ||
                   MPI_IN_PLACE == recvbuf) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_ARG,
                                          FUNC_NAME);
        } else {
            if(MPI_IN_PLACE != sendbuf) {
                OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, sendcount);
                OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
            }
            OMPI_CHECK_DATATYPE_FOR_RECV(err, recvtype, recvcount);
            OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
        }

        if (MPI_IN_PLACE != sendbuf && !OMPI_COMM_IS_INTER(comm)) {
            size_t sendtype_size, recvtype_size_tmp;
            ompi_datatype_type_size(sendtype, &sendtype_size);
            ompi_datatype_type_size(recvtype, &recvtype_size_tmp);
            if ((sendtype_size*sendcount) != (recvtype_size_tmp*recvcount)) {
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TRUNCATE, FUNC_NAME);
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

    if (! OMPI_COMM_IS_INTER(comm)) {
        ompi_datatype_type_size(recvtype, &recvtype_size);
        if( (0 == recvcount) || (0 == recvtype_size) ) {
            return MPI_SUCCESS;
        }
    }

    /* Invoke the coll component to perform the back-end operation */
#ifndef ENABLE_ANALYSIS
    err = comm->c_coll->coll_alltoall(sendbuf, sendcount, sendtype,
                                     recvbuf, recvcount, recvtype,
                                     comm, comm->c_coll->coll_alltoall_module);
#else
    err = comm->c_coll->coll_alltoall(sendbuf, sendcount, sendtype,
                                     recvbuf, recvcount, recvtype,
                                     comm, comm->c_coll->coll_alltoall_module, &item);
    qentryIntoQueue(&item);
#endif
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}

