/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013-2018 Inria.  All rights reserved.
 * Copyright (c) 2019      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "pml_monitoring.h"

int mca_pml_monitoring_isend_init(const void *buf,
                                  size_t count,
                                  ompi_datatype_t *datatype,
                                  int dst,
                                  int tag,
                                  mca_pml_base_send_mode_t mode,
                                  struct ompi_communicator_t* comm,
                                  struct ompi_request_t **request
#ifdef ENABLE_ANALYSIS
                                  , qentry **q
#endif
                                  )
{
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
    return pml_selected_module.pml_isend_init(buf, count, datatype,
                                              dst, tag, mode, comm, request, &item);
#else
    return pml_selected_module.pml_isend_init(buf, count, datatype,
                                              dst, tag, mode, comm, request);
#endif
}

int mca_pml_monitoring_isend(const void *buf,
                             size_t count,
                             ompi_datatype_t *datatype,
                             int dst,
                             int tag,
                             mca_pml_base_send_mode_t mode,
                             struct ompi_communicator_t* comm,
                             struct ompi_request_t **request
#ifdef ENABLE_ANALYSIS
                             , qentry **q
#endif
                             )
{
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    int world_rank;
    /**
     * If this fails the destination is not part of my MPI_COM_WORLD
     * Lookup its name in the rank hashtable to get its MPI_COMM_WORLD rank
     */
    if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(dst, comm->c_remote_group, &world_rank)) {
        size_t type_size, data_size;
        ompi_datatype_type_size(datatype, &type_size);
        data_size = count*type_size;
        mca_common_monitoring_record_pml(world_rank, data_size, tag);
    }

#ifndef ENABLE_ANALYSIS
    return pml_selected_module.pml_isend(buf, count, datatype,
                                         dst, tag, mode, comm, request);
#else
    return pml_selected_module.pml_isend(buf, count, datatype,
                                         dst, tag, mode, comm, request, &item);
#endif
}

int mca_pml_monitoring_send(const void *buf,
                            size_t count,
                            ompi_datatype_t *datatype,
                            int dst,
                            int tag,
                            mca_pml_base_send_mode_t mode,
                            struct ompi_communicator_t* comm
#ifdef ENABLE_ANALYSIS
                             , qentry **q
#endif
                            )
{
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    int world_rank;
    /* Are we sending to a peer from my own MPI_COMM_WORLD? */
    if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(dst, comm->c_remote_group, &world_rank)) {
        size_t type_size, data_size;
        ompi_datatype_type_size(datatype, &type_size);
        data_size = count*type_size;
        mca_common_monitoring_record_pml(world_rank, data_size, tag);
    }

#ifndef ENABLE_ANALYSIS
    return pml_selected_module.pml_send(buf, count, datatype,
                                        dst, tag, mode, comm);
#else
    return pml_selected_module.pml_send(buf, count, datatype,
                                        dst, tag, mode, comm, &item);
#endif
}
