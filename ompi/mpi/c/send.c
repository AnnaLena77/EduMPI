/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015      Research Organization for Information Science
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
#include "ompi/mca/pml/pml.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/memchecker.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/mpi/c/init.h"
#include <time.h>

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Send = PMPI_Send
#endif
#define MPI_Send PMPI_Send
#endif

static const char FUNC_NAME[] = "MPI_Send";


int MPI_Send(const void *buf, int count, MPI_Datatype type, int dest,
             int tag, MPI_Comm comm)
{
    #ifdef ENABLE_ANALYSIS
    //printf("Hallo aus dem send \n");
    time_t current_time = time(NULL);
    char *operation = "send";
    char *comm_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int comm_name_length;
    MPI_Comm_get_name(comm, comm_name, &comm_name_length);
    char *type_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int type_name_length;
    MPI_Type_get_name(type, type_name, &type_name_length);
    int processrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processrank);
    enqueue(&operation, &type_name, count, count*sizeof(type), &comm_name, processrank, dest, current_time);
    #endif
    int rc = MPI_SUCCESS;
    
    //OMPI_SPC in ompi_spc.h
    SPC_RECORD(OMPI_SPC_SEND, 1);
    
    //MEMCHECKER = Runtime-Memory-Checker (OPAL-Framework)
    /* The Memchecker-MCA is implemented to allow MPI-semantic checking for    your application (as well as internals of Open MPI), with the help of memory checking tools such as the Memcheck of the Valgrind-suite */
    MEMCHECKER(
        memchecker_datatype(type);
        memchecker_call(&opal_memchecker_base_isdefined, buf, count, type);
        memchecker_comm(comm);
    );
    
    //the parameters to each MPI function can be passed through a series of correctness checks
    if ( MPI_PARAM_CHECK ) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        //Check nach Communicator
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM, FUNC_NAME);
          //check, ob count (Anzahl gesendeter Objekte kleiner 0 ist), dann Error
        } else if (count < 0) {
            rc = MPI_ERR_COUNT;
          //
        } else if (tag < 0 || tag > mca_pml.pml_max_tag) {
            rc = MPI_ERR_TAG;
        } else if (ompi_comm_peer_invalid(comm, dest) &&
                   (MPI_PROC_NULL != dest)) {
            rc = MPI_ERR_RANK;
        } else {
            OMPI_CHECK_DATATYPE_FOR_SEND(rc, type, count);
            OMPI_CHECK_USER_BUFFER(rc, buf, type, count);
        }
        OMPI_ERRHANDLER_CHECK(rc, comm, rc, FUNC_NAME);
    }

    if (MPI_PROC_NULL == dest) {
        return MPI_SUCCESS;
    }
    
    //printf("Hier wird gesendet\n");
    
    //MCA_PML_CALL is a macro from pml.h!
    rc = MCA_PML_CALL(send(buf, count, type, dest, tag, MCA_PML_BASE_SEND_STANDARD, comm));
    //printf("Sender rc: %d\n", rc);
    OMPI_ERRHANDLER_RETURN(rc, comm, rc, FUNC_NAME);
}
