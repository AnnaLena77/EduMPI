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
 * Copyright (c) 2015      Los Alamos National Security, LLC.  All rights
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
#include "ompi/win/win.h"
#include "ompi/mca/osc/osc.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Get = PMPI_Get
#endif
#define MPI_Get PMPI_Get
#endif

static const char FUNC_NAME[] = "MPI_Get";

//Daten aus Window des Target-Prozesses abrufen
int MPI_Get(void *origin_addr, int origin_count,
            MPI_Datatype origin_datatype, int target_rank,
            MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win)
{

#ifdef ENABLE_ANALYSIS
    qentry *item = (qentry*)malloc(sizeof(qentry));
    initQentry(&item);
    item->start = time(NULL);
    //Basic information
    strcpy(item->function, "MPI_Get");
    strcpy(item->communicationType, "one-sided");
    item->blocking = 0; //One-Sided-Communication is always non-blocking!
    /*Datatype --> if there is a difference between the origin_datatype and the target_datatype, 
    write both into database*/
    char *origin_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int origin_name_length;
    MPI_Type_get_name(origin_datatype, origin_name, &origin_name_length);
    char *target_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int target_name_length;
    MPI_Type_get_name(target_datatype, target_name, &target_name_length);
    if(strcmp(origin_name, target_name)==0){
        strcpy(item->datatype, origin_name);
        free(origin_name);
        free(target_name);
    }
    else {
        strcat(origin_name, ", ");
        strcat(origin_name, target_name);
        strcpy(item->datatype, origin_name);
        free(origin_name);
        free(target_name);
    }
    //count and datasize
    item->count = origin_count + target_count;
    item->datasize = origin_count*sizeof(origin_datatype)+target_count*sizeof(target_datatype);
    //Name of the window instead of the communicator
    char *window_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int window_name_length;
    MPI_Win_get_name(win, window_name, &window_name_length);
    strcpy(item->communicationArea, window_name);
    free(window_name);
    
    MPI_Group wingroup;
    MPI_Win_get_group(win, &wingroup);
    //processrank and partnerrank
    int processrank;
    MPI_Group_rank(wingroup, &processrank);
    item->processrank = processrank;
    item->partnerrank = target_rank;
    
    MPI_Group_free(&wingroup);
#endif 

    int rc;

    SPC_RECORD(OMPI_SPC_GET, 1);

    if (MPI_PARAM_CHECK) {
        rc = OMPI_SUCCESS;

        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);

        if (ompi_win_invalid(win)) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_WIN, FUNC_NAME);
        } else if (origin_count < 0 || target_count < 0) {
            rc = MPI_ERR_COUNT;
        } else if (ompi_win_peer_invalid(win, target_rank) &&
                   (MPI_PROC_NULL != target_rank)) {
            rc = MPI_ERR_RANK;
        } else if ( MPI_WIN_FLAVOR_DYNAMIC != win->w_flavor && target_disp < 0 ) {
            rc = MPI_ERR_DISP;
        } else {
            OMPI_CHECK_DATATYPE_FOR_ONE_SIDED(rc, origin_datatype, origin_count);
            if (OMPI_SUCCESS == rc) {
                OMPI_CHECK_DATATYPE_FOR_ONE_SIDED(rc, target_datatype, target_count);
            }
        }
        OMPI_ERRHANDLER_CHECK(rc, win, rc, FUNC_NAME);
    }

    if (MPI_PROC_NULL == target_rank) return MPI_SUCCESS;

#ifndef ENABLE_ANALYSIS
    rc = win->w_osc_module->osc_get(origin_addr, origin_count, origin_datatype,
                                    target_rank, target_disp, target_count,
                                    target_datatype, win);
#else
        rc = win->w_osc_module->osc_get(origin_addr, origin_count, origin_datatype,
                                    target_rank, target_disp, target_count,
                                    target_datatype, win, &item);
        qentryIntoQueue(&item);
#endif
    OMPI_ERRHANDLER_RETURN(rc, win, rc, FUNC_NAME);
}

