/*
 * Copyright (c) 2016-2018 Inria. All rights reserved.
 * Copyright (c) 2019      Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "ompi/request/request.h"
#include "ompi/communicator/communicator.h"
#include "coll_monitoring.h"

int mca_coll_monitoring_barrier(struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module)
{
    mca_coll_monitoring_module_t*monitoring_module = (mca_coll_monitoring_module_t*) module;
    int i, rank;
    const int comm_size = ompi_comm_size(comm);
    const int my_rank = ompi_comm_rank(comm);
    for( i = 0; i < comm_size; ++i ) {
	if( my_rank == i ) continue; /* No communication for self */
	/**
	 * If this fails the destination is not part of my MPI_COM_WORLD
	 * Lookup its name in the rank hashtable to get its MPI_COMM_WORLD rank
	 */
	if( OPAL_SUCCESS == mca_common_monitoring_get_world_rank(i, comm->c_remote_group, &rank) ) {
	    mca_common_monitoring_record_coll(rank, 0);
	}
    }
    mca_common_monitoring_coll_a2a(0, monitoring_module->data);
#ifndef ENABLE_ANALYSIS
    return monitoring_module->real.coll_barrier(comm, monitoring_module->real.coll_barrier_module);
#else
    return monitoring_module->real.coll_barrier(comm, monitoring_module->real.coll_barrier_module, NULL);
#endif
}

int mca_coll_monitoring_ibarrier(struct ompi_communicator_t *comm,
                                 ompi_request_t ** request,
                                 mca_coll_base_module_t *module)
{
    mca_coll_monitoring_module_t*monitoring_module = (mca_coll_monitoring_module_t*) module;
    int i, rank;
    const int comm_size = ompi_comm_size(comm);
    const int my_rank = ompi_comm_rank(comm);
    for( i = 0; i < comm_size; ++i ) {
	if( my_rank == i ) continue; /* No communication for self */
	/**
	 * If this fails the destination is not part of my MPI_COM_WORLD
	 * Lookup its name in the rank hashtable to get its MPI_COMM_WORLD rank
	 */
	if( OPAL_SUCCESS == mca_common_monitoring_get_world_rank(i, comm->c_remote_group, &rank) ) {
	    mca_common_monitoring_record_coll(rank, 0);
	}
    }
    mca_common_monitoring_coll_a2a(0, monitoring_module->data);
#ifndef ENABLE_ANALYSIS
    return monitoring_module->real.coll_ibarrier(comm, request, monitoring_module->real.coll_ibarrier_module);
#else
    return monitoring_module->real.coll_ibarrier(comm, request, monitoring_module->real.coll_ibarrier_module, NULL);
#endif
}
