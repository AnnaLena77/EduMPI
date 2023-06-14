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
 * Copyright (c) 2010      University of Houston.  All rights reserved.
 * Copyright (c) 2010-2012 Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
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

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Allgather = PMPI_Allgather
#endif
#define MPI_Allgather PMPI_Allgather
#endif

static const char FUNC_NAME[] = "MPI_Allgather";

//Jeder Prozess versendet ein Element und jeder Prozess empfängt gleichzeitig das gesendete Element aller anderen Prozesse
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
#ifdef ENABLE_ANALYSIS
    qentry *item = (qentry*)malloc(sizeof(qentry));
    initQentry(&item);
    gettimeofday(&item->start, NULL);
    strcpy(item->function, "MPI_Allgather");
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

    //item->count
    int number_of_processes = ompi_comm_size(comm);
    //item->count = sendcount+recvcount;
    //item->datasize
    //item->datasize = sendcount * sizeof(sendtype) + recvcount * sizeof(recvtype);
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
    
    //item->processorname
    char *proc_name = (char*)malloc(MPI_MAX_PROCESSOR_NAME);
    int proc_name_length;
    MPI_Get_processor_name(proc_name, &proc_name_length);
    strcpy(item->processorname, proc_name);
    free(proc_name);
#endif 
    int err;

    SPC_RECORD(OMPI_SPC_ALLGATHER, 1);

    MEMCHECKER(
        int rank;
        ptrdiff_t ext;

        rank = ompi_comm_rank(comm);
        ompi_datatype_type_extent(recvtype, &ext);

        memchecker_datatype(recvtype);
        memchecker_comm(comm);
        /* check whether the actual send buffer is defined. */
        if (MPI_IN_PLACE == sendbuf) {
            memchecker_call(&opal_memchecker_base_isdefined,
                            (char *)(recvbuf)+rank*recvcount*ext,
                            recvcount, recvtype);
        } else {
            memchecker_datatype(sendtype);
            memchecker_call(&opal_memchecker_base_isdefined, sendbuf, sendcount, sendtype);
        }
        /* check whether the receive buffer is addressable. */
        memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, recvcount, recvtype);
    );

    if (MPI_PARAM_CHECK) {

        /* Unrooted operation -- same checks for all ranks on both
           intracommunicators and intercommunicators */

        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
          OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_COMM, FUNC_NAME);
        } else if (MPI_DATATYPE_NULL == recvtype || NULL == recvtype) {
          err = MPI_ERR_TYPE;
        } else if (recvcount < 0) {
          err = MPI_ERR_COUNT;
        } else if ((MPI_IN_PLACE == sendbuf && OMPI_COMM_IS_INTER(comm)) ||
                   MPI_IN_PLACE == recvbuf) {
          return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, FUNC_NAME);
        } else if (MPI_IN_PLACE != sendbuf) {
            OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, sendcount);
        }
        OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
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

    /* Do we need to do anything?  Everyone had to give the same send
       signature, which means that everyone must have given a
       sendcount > 0 if there's anything to send for the intra-communicator
       case.  If we're doing IN_PLACE, however, check recvcount,
       not sendcount. */
    if ( OMPI_COMM_IS_INTRA(comm) ) {
       if ((MPI_IN_PLACE != sendbuf && 0 == sendcount) ||
            (0 == recvcount)) {
            return MPI_SUCCESS;
	   }
    }
    else if ( OMPI_COMM_IS_INTER(comm) ){
        /* for inter comunicators, the communication pattern
           need not be symmetric. Specifically, one group is
	   allows to have sendcount=0, while the other has
           a valid sendcount. Thus, the only way not to do
           anything is if both sendcount and recvcount are zero. */
	    if ( 0 == sendcount && 0 == recvcount ) {
	        return MPI_SUCCESS;
	    }
    }

    /* Invoke the coll component to perform the back-end operation */
#ifndef ENABLE_ANALYSIS
    err = comm->c_coll->coll_allgather(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcount, recvtype, comm,
                                      comm->c_coll->coll_allgather_module);
#else
    err = comm->c_coll->coll_allgather(sendbuf, sendcount, sendtype,
                                      recvbuf, recvcount, recvtype, comm,
                                      comm->c_coll->coll_allgather_module, &item);
    qentryIntoQueue(&item);
#endif
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}

