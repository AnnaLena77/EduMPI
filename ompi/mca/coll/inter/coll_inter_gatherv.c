/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2010 University of Houston. All rights reserved.
 * Copyright (c) 2015-2016 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_inter.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"

/*
 *	gatherv_inter
 *
 *	Function:	- gatherv operation using a local gather on c_local_comm
 *	Accepts:	- same arguments as MPI_Gatherv()
 *	Returns:	- MPI_SUCCESS or error code
 */
int
mca_coll_inter_gatherv_inter(const void *sbuf, int scount,
                             struct ompi_datatype_t *sdtype,
                             void *rbuf, const int *rcounts, const int *disps,
                             struct ompi_datatype_t *rdtype, int root,
                             struct ompi_communicator_t *comm,
                             mca_coll_base_module_t *module
#ifdef ENABLE_ANALYSIS
			  , qentry **q
#endif
                             )
{
#ifdef ENABLE_ANALYSIS
     qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    }
    else item = NULL;
#endif
    int i, rank, size, size_local, err;
    size_t total = 0;
    int *count=NULL, *displace=NULL;
    char *ptmp_free=NULL, *ptmp=NULL;
    ompi_datatype_t *ndtype;

    if (MPI_PROC_NULL == root) { /* do nothing */
        return OMPI_SUCCESS;
    }
    size = ompi_comm_remote_size(comm);
    rank = ompi_comm_rank(comm);
    size_local = ompi_comm_size(comm);

    if (MPI_ROOT == root) { /* I am the root, receiving the data from zero. */
        ompi_datatype_create_indexed(size, rcounts, disps, rdtype, &ndtype);
        ompi_datatype_commit(&ndtype);
#ifndef ENABLE_ANALYSIS
        err = MCA_PML_CALL(recv(rbuf, 1, ndtype, 0,
                                MCA_COLL_BASE_TAG_GATHERV,
                                comm, MPI_STATUS_IGNORE));
#else
        err = MCA_PML_CALL(recv(rbuf, 1, ndtype, 0,
                                MCA_COLL_BASE_TAG_GATHERV,
                                comm, MPI_STATUS_IGNORE, &item));
#endif
        ompi_datatype_destroy(&ndtype);
        return err;
    }

    if (0 == rank) {
        count = (int *)malloc(sizeof(int) * size_local);
        displace = (int *)malloc(sizeof(int) * size_local);
        if ((NULL == displace) || (NULL == count)) {
            err = OMPI_ERR_OUT_OF_RESOURCE;
            goto exit;
        }
    }

#ifndef ENABLE_ANALYSIS
    err = comm->c_local_comm->c_coll->coll_gather(&scount, 1, MPI_INT,
                                                 count, 1, MPI_INT,
                                                 0, comm->c_local_comm,
                                                 comm->c_local_comm->c_coll->coll_gather_module);
#else
    err = comm->c_local_comm->c_coll->coll_gather(&scount, 1, MPI_INT,
                                                 count, 1, MPI_INT,
                                                 0, comm->c_local_comm,
                                                 comm->c_local_comm->c_coll->coll_gather_module, &item);
#endif
    if (OMPI_SUCCESS != err) {
        goto exit;
    }
    if(0 == rank) {
        displace[0] = 0;
        for (i = 1; i < size_local; i++) {
            displace[i] = displace[i-1] + count[i-1];
        }
        /* Perform the gatherv locally with the first process as root */
        for (i = 0; i < size_local; i++) {
            total = total + count[i];
        }
        if ( total > 0 ) {
            ptrdiff_t gap, span;
            span = opal_datatype_span(&sdtype->super, total, &gap);
            ptmp_free = (char*)malloc(span);
            if (NULL == ptmp_free) {
                err = OMPI_ERR_OUT_OF_RESOURCE;
                goto exit;
            }
            ptmp = ptmp_free - gap;
        }
    }
#ifndef ENABLE_ANALYSIS
    err = comm->c_local_comm->c_coll->coll_gatherv(sbuf, scount, sdtype,
                                                  ptmp, count, displace,
                                                  sdtype,0, comm->c_local_comm,
                                                  comm->c_local_comm->c_coll->coll_gatherv_module);
#else
    err = comm->c_local_comm->c_coll->coll_gatherv(sbuf, scount, sdtype,
                                                  ptmp, count, displace,
                                                  sdtype,0, comm->c_local_comm,
                                                  comm->c_local_comm->c_coll->coll_gatherv_module, &item);
#endif
    if (OMPI_SUCCESS != err) {
        goto exit;
    }

    if (0 == rank) {
        /* First process sends data to the root */
#ifndef ENABLE_ANALYSIS
        err = MCA_PML_CALL(send(ptmp, total, sdtype, root,
                                MCA_COLL_BASE_TAG_GATHERV,
                                MCA_PML_BASE_SEND_STANDARD, comm));
#else
        err = MCA_PML_CALL(send(ptmp, total, sdtype, root,
                                MCA_COLL_BASE_TAG_GATHERV,
                                MCA_PML_BASE_SEND_STANDARD, comm, &item));
#endif
    }

  exit:
    if (NULL != ptmp_free) {
        free(ptmp_free);
    }
    if (NULL != displace) {
        free(displace);
    }
    if (NULL != count) {
        free(count);
    }

    /* All done */
    return err;
}
