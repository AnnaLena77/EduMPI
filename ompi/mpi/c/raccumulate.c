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
 * Copyright (c) 2009      Sun Microsystmes, Inc.  All rights reserved.
 * Copyright (c) 2011      Sandia National Laboratories. All rights reserved.
 * Copyright (c) 2014-2015 Los Alamos National Security, LLC. All rights
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
#include "ompi/op/op.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/datatype/ompi_datatype_internal.h"
#include "ompi/memchecker.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Raccumulate = PMPI_Raccumulate
#endif
#define MPI_Raccumulate PMPI_Raccumulate
#endif

static const char FUNC_NAME[] = "MPI_Raccumulate";

//Request-Handler ermoelicht Synchronisation ueber MPI_Wait
int MPI_Raccumulate(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
                   int target_rank, MPI_Aint target_disp, int target_count,
                   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request *request)
{
#ifdef ENABLE_ANALYSIS
    qentry *item = (qentry*)malloc(sizeof(qentry));
    initQentry(&item);
    item->start = time(NULL);
    //Basic information
    strcpy(item->function, "MPI_Raccumulate");
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
    item->count = origin_count;
    item->datasize = origin_count*sizeof(origin_datatype);
    //operation
    strcpy(item->operation, op->o_name);
    //Name of communicator
    char *comm_name = (char*) malloc(MPI_MAX_OBJECT_NAME);
    int comm_name_length;
    ompi_win_get_communicator(win, comm_name, &comm_name_length);
    strcpy(item->communicationArea, comm_name);
    free(comm_name);
    
    MPI_Group wingroup;
    MPI_Win_get_group(win, &wingroup);
    //operation
    strcpy(item->operation, op->o_name);
    //processrank and partnerrank
    int processrank;
    MPI_Group_rank(wingroup, &processrank);
    item->processrank = processrank;
    item->partnerrank = target_rank;
    
    MPI_Group_free(&wingroup);
#endif  
    int rc;
    ompi_win_t *ompi_win = (ompi_win_t*) win;

    MEMCHECKER(
        memchecker_datatype(origin_datatype);
        memchecker_datatype(target_datatype);
        memchecker_call(&opal_memchecker_base_isdefined, origin_addr, origin_count, origin_datatype);
    );

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
        } else if (MPI_OP_NULL == op || MPI_NO_OP == op) {
            rc = MPI_ERR_OP;
        } else if (!ompi_op_is_intrinsic(op)) {
            rc = MPI_ERR_OP;
        } else if ( MPI_WIN_FLAVOR_DYNAMIC != win->w_flavor && target_disp < 0 ) {
            rc = MPI_ERR_DISP;
        } else {
            OMPI_CHECK_DATATYPE_FOR_ONE_SIDED(rc, origin_datatype, origin_count);
            if (OMPI_SUCCESS == rc) {
                OMPI_CHECK_DATATYPE_FOR_ONE_SIDED(rc, target_datatype, target_count);
            }
            if (OMPI_SUCCESS == rc) {
                /* While technically the standard probably requires that the
                   datatypes used with MPI_REPLACE conform to all the rules
                   for other reduction operators, we don't require such
                   behavior, as checking for it is expensive here and we don't
                   care in implementation.. */
                if (op != &ompi_mpi_op_replace.op && op != &ompi_mpi_op_no_op.op) {
                    ompi_datatype_t *op_check_dt, *origin_check_dt;
                    char *msg;

                    /* RACCUMULATE, unlike REDUCE, can use with derived
                       datatypes with predefinied operations, with some
                       restrictions outlined in MPI-3:11.3.4.  The derived
                       datatype must be composed entierly from one predefined
                       datatype (so you can do all the construction you want,
                       but at the bottom, you can only use one datatype, say,
                       MPI_INT).  If the datatype at the target isn't
                       predefined, then make sure it's composed of only one
                       datatype, and check that datatype against
                       ompi_op_is_valid(). */
                    origin_check_dt = ompi_datatype_get_single_predefined_type_from_args(origin_datatype);
                    op_check_dt = ompi_datatype_get_single_predefined_type_from_args(target_datatype);

                    if( !((origin_check_dt == op_check_dt) & (NULL != op_check_dt)) ) {
                        OMPI_ERRHANDLER_RETURN(MPI_ERR_ARG, win, MPI_ERR_ARG, FUNC_NAME);
                    }

                    /* check to make sure primitive type is valid for
                       reduction.  Should do this on the target, but
                       then can't get the errcode back for this
                       call */
                    if (!ompi_op_is_valid(op, op_check_dt, &msg, FUNC_NAME)) {
                        int ret = OMPI_ERRHANDLER_INVOKE(win, MPI_ERR_OP, msg);
                        free(msg);
                        return ret;
                    }
                }
            }
        }
        OMPI_ERRHANDLER_CHECK(rc, win, rc, FUNC_NAME);
    }

    if (MPI_PROC_NULL == target_rank) {
        *request = &ompi_request_empty;
        return MPI_SUCCESS;
    }

#ifndef ENABLE_ANALYSIS
    rc = ompi_win->w_osc_module->osc_raccumulate(origin_addr,
                                                origin_count,
                                                origin_datatype,
                                                target_rank,
                                                target_disp,
                                                target_count,
                                                target_datatype,
                                                op, win, request);
#else
    rc = ompi_win->w_osc_module->osc_raccumulate(origin_addr,
                                                origin_count,
                                                origin_datatype,
                                                target_rank,
                                                target_disp,
                                                target_count,
                                                target_datatype,
                                                op, win, request, &item);
    qentryIntoQueue(&item);
#endif
    OMPI_ERRHANDLER_RETURN(rc, win, rc, FUNC_NAME);
}
