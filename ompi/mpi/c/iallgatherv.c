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
 * Copyright (c) 2010      University of Houston.  All rights reserved.
 * Copyright (c) 2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012-2013 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/memchecker.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Iallgatherv = PMPI_Iallgatherv
#endif
#define MPI_Iallgatherv PMPI_Iallgatherv
#endif

static const char FUNC_NAME[] = "MPI_Iallgatherv";


int MPI_Iallgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, const int recvcounts[], const int displs[],
                    MPI_Datatype recvtype, MPI_Comm comm, MPI_Request *request)
{
#ifdef ENABLE_ANALYSIS
    /*qentry *item = (qentry*)malloc(sizeof(qentry));
    initQentry(&item);
    gettimeofday(&item->start, NULL);*/
    
    qentry *item = getWritingRingPos();
    initQentry(&item);
    //item->start
    clock_gettime(CLOCK_REALTIME, &item->start);
    
    strcpy(item->function, "MPI_Allgatherv");
    strcpy(item->communicationType, "collective");
    //item->datatype
    char *sendtype_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int sendtype_name_length;
    MPI_Type_get_name(sendtype, sendtype_name, &sendtype_name_length);
    char *recvtype_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int recvtype_name_length;
    MPI_Type_get_name(recvtype, recvtype_name, &recvtype_name_length);
    if(strcmp(sendtype_name, recvtype_name)==0){
        strcpy(item->datatype, sendtype_name);
        free(sendtype_name);
        free(recvtype_name);
    }
    else {
        strcat(sendtype_name, ", ");
        strcat(sendtype_name, recvtype_name);
        strcpy(item->datatype, sendtype_name);
        free(sendtype_name);
        free(recvtype_name);
    }

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


    item->blocking = 0;
    
    //item->processorname
    char *proc_name = (char*)malloc(MPI_MAX_PROCESSOR_NAME);
    int proc_name_length;
    MPI_Get_processor_name(proc_name, &proc_name_length);
    strcpy(item->processorname, proc_name);
    free(proc_name);
    
#endif 
    int i, size, err;

    SPC_RECORD(OMPI_SPC_IALLGATHERV, 1);

    MEMCHECKER(
        int rank;
        ptrdiff_t ext;

        rank = ompi_comm_rank(comm);
        size = ompi_comm_size(comm);
        ompi_datatype_type_extent(recvtype, &ext);

        memchecker_datatype(recvtype);
        memchecker_comm (comm);
        /* check whether the receive buffer is addressable. */
        for (i = 0; i < size; i++) {
            memchecker_call(&opal_memchecker_base_isaddressable,
                            (char *)(recvbuf)+displs[i]*ext,
                            recvcounts[i], recvtype);
        }

        /* check whether the actual send buffer is defined. */
        if (MPI_IN_PLACE == sendbuf) {
            memchecker_call(&opal_memchecker_base_isdefined,
                            (char *)(recvbuf)+displs[rank]*ext,
                            recvcounts[rank], recvtype);
        } else {
            memchecker_datatype(sendtype);
            memchecker_call(&opal_memchecker_base_isdefined, sendbuf, sendcount, sendtype);
        }
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
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, FUNC_NAME);
        } else if (MPI_DATATYPE_NULL == recvtype) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TYPE, FUNC_NAME);
        }

        if (MPI_IN_PLACE != sendbuf) {
            OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, sendcount);
        }
        OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);

      /* We always define the remote group to be the same as the local
         group in the case of an intracommunicator, so it's safe to
         get the size of the remote group here for both intra- and
         intercommunicators */

        size = ompi_comm_remote_size(comm);
        for (i = 0; i < size; ++i) {
          if (recvcounts[i] < 0) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_COUNT, FUNC_NAME);
          }
        }

        if (NULL == displs) {
          return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_BUFFER, FUNC_NAME);
        }
    }

    /* Invoke the coll component to perform the back-end operation */
#ifndef ENABLE_ANALYSIS
    err = comm->c_coll->coll_iallgatherv(sendbuf, sendcount, sendtype,
                                        recvbuf, recvcounts, displs,
                                        recvtype, comm, request,
                                        comm->c_coll->coll_iallgatherv_module);
#else
    err = comm->c_coll->coll_iallgatherv(sendbuf, sendcount, sendtype,
                                        recvbuf, recvcounts, displs,
                                        recvtype, comm, request,
                                        comm->c_coll->coll_iallgatherv_module, &item);
#endif
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ompi_coll_base_retain_datatypes(*request, (MPI_IN_PLACE==sendbuf)?NULL:sendtype, recvtype);
    }
#ifdef ENABLE_ANALYSIS
    clock_gettime(CLOCK_REALTIME, &item->end);
    //qentryIntoQueue(&item);
#endif
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}

