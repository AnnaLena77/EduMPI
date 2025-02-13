/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2020      Amazon.com, Inc. or its affiliates.
 *                         All Rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/base.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "coll_tuned.h"

/*
 * Notes on evaluation rules and ordering
 *
 * The order is:
 *      use file based rules if presented (-coll_tuned_dynamic_rules_filename = rules)
 * Else
 *      use forced rules (-coll_tuned_dynamic_ALG_intra_algorithm = algorithm-number)
 * Else
 *      use fixed (compiled) rule set (or nested ifs)
 *
 */

/*
 *  allreduce_intra
 *
 *  Function:   - allreduce using other MPI collectives
 *  Accepts:    - same as MPI_Allreduce()
 *  Returns:    - MPI_SUCCESS or error code
 */
int
ompi_coll_tuned_allreduce_intra_dec_dynamic (const void *sbuf, void *rbuf, int count,
                                             struct ompi_datatype_t *dtype,
                                             struct ompi_op_t *op,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif

    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_allreduce_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLREDUCE].algorithm) {

#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_allreduce_intra_do_this(sbuf, rbuf, count, dtype, op, comm, module,
                                                       tuned_module->user_forced[ALLREDUCE].algorithm,
                                                       tuned_module->user_forced[ALLREDUCE].tree_fanout,
                                                       tuned_module->user_forced[ALLREDUCE].segsize);
#else
        if(item!=NULL){
            int algo = tuned_module->user_forced[ALLREDUCE].algorithm;
            if(algo == 1) memcpy(item->usedAlgorithm, "basic_linear", 12);
            else if(algo == 2) memcpy(item->usedAlgorithm, "nonoverlapping", 14);
            else if(algo == 3) memcpy(item->usedAlgorithm, "recursive_doubling", 18);
            else if(algo == 4) memcpy(item->usedAlgorithm, "ring", 4);
            else if(algo == 5) memcpy(item->usedAlgorithm, "segmented_ring", 14);
            else if(algo == 6) memcpy(item->usedAlgorithm, "rabenseifner", 12);
            else if(algo == 7) memcpy(item->usedAlgorithm, "allgather_reduce", 16);
        }
        return ompi_coll_tuned_allreduce_intra_do_this(sbuf, rbuf, count, dtype, op, comm, module,
                                                       tuned_module->user_forced[ALLREDUCE].algorithm,
                                                       tuned_module->user_forced[ALLREDUCE].tree_fanout,
                                                       tuned_module->user_forced[ALLREDUCE].segsize, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[ALLREDUCE]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLREDUCE],
                                                        dsize, &faninout, &segsize, &ignoreme);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_allreduce_intra_do_this (sbuf, rbuf, count, dtype, op,
                                                            comm, module,
                                                            alg, faninout, segsize);
#else
            if(item!=NULL){
                int algo = alg;
                if(algo == 1) memcpy(item->usedAlgorithm, "basic_linear", 12);
                else if(algo == 2) memcpy(item->usedAlgorithm, "nonoverlapping", 14);
                else if(algo == 3) memcpy(item->usedAlgorithm, "recursive_doubling", 18);
                else if(algo == 4) memcpy(item->usedAlgorithm, "ring", 4);
                else if(algo == 5) memcpy(item->usedAlgorithm, "segmented_ring", 14);
                else if(algo == 6) memcpy(item->usedAlgorithm, "rabenseifner", 12);
                else if(algo == 7) memcpy(item->usedAlgorithm, "allgather_reduce", 16);
            }
            return ompi_coll_tuned_allreduce_intra_do_this (sbuf, rbuf, count, dtype, op,
                                                            comm, module,
                                                            alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_allreduce_intra_dec_fixed (sbuf, rbuf, count, dtype, op,
                                                      comm, module);
#else
    return ompi_coll_tuned_allreduce_intra_dec_fixed (sbuf, rbuf, count, dtype, op,
                                                      comm, module, &item);
#endif
}

/*
 *    alltoall_intra_dec
 *
 *    Function:    - seletects alltoall algorithm to use
 *    Accepts:    - same arguments as MPI_Alltoall()
 *    Returns:    - MPI_SUCCESS or error code (passed from the alltoall implementation)
 */

int ompi_coll_tuned_alltoall_intra_dec_dynamic(const void *sbuf, int scount,
                                               struct ompi_datatype_t *sdtype,
                                               void* rbuf, int rcount,
                                               struct ompi_datatype_t *rdtype,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif

    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_alltoall_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLTOALL].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_alltoall_intra_do_this(sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module,
                                                      tuned_module->user_forced[ALLTOALL].algorithm,
                                                      tuned_module->user_forced[ALLTOALL].tree_fanout,
                                                      tuned_module->user_forced[ALLTOALL].segsize,
                                                      tuned_module->user_forced[ALLTOALL].max_requests);
#else
        return ompi_coll_tuned_alltoall_intra_do_this(sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module,
                                                      tuned_module->user_forced[ALLTOALL].algorithm,
                                                      tuned_module->user_forced[ALLTOALL].tree_fanout,
                                                      tuned_module->user_forced[ALLTOALL].segsize,
                                                      tuned_module->user_forced[ALLTOALL].max_requests, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[ALLTOALL]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int comsize;
        int alg, faninout, segsize, max_requests;
        size_t dsize;

        ompi_datatype_type_size (sdtype, &dsize);
        comsize = ompi_comm_size(comm);
        dsize *= (ptrdiff_t)comsize * (ptrdiff_t)scount;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLTOALL],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_alltoall_intra_do_this (sbuf, scount, sdtype,
                                                           rbuf, rcount, rdtype,
                                                           comm, module,
                                                           alg, faninout, segsize, max_requests);
#else
            return ompi_coll_tuned_alltoall_intra_do_this (sbuf, scount, sdtype,
                                                           rbuf, rcount, rdtype,
                                                           comm, module,
                                                           alg, faninout, segsize, max_requests, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_alltoall_intra_dec_fixed (sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     comm, module);
#else
    return ompi_coll_tuned_alltoall_intra_dec_fixed (sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     comm, module, &item);
#endif
}

/*
 *    Function:   - selects alltoallv algorithm to use
 *    Accepts:    - same arguments as MPI_Alltoallv()
 *    Returns:    - MPI_SUCCESS or error code
 */

int ompi_coll_tuned_alltoallv_intra_dec_dynamic(const void *sbuf, const int *scounts, const int *sdisps,
                                                struct ompi_datatype_t *sdtype,
                                                void* rbuf, const int *rcounts, const int *rdisps,
                                                struct ompi_datatype_t *rdtype,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "ompi_coll_tuned_alltoallv_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLTOALLV].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_alltoallv_intra_do_this(sbuf, scounts, sdisps, sdtype,
                                                       rbuf, rcounts, rdisps, rdtype,
                                                       comm, module,
                                                       tuned_module->user_forced[ALLTOALLV].algorithm);
#else
        return ompi_coll_tuned_alltoallv_intra_do_this(sbuf, scounts, sdisps, sdtype,
                                                       rbuf, rcounts, rdisps, rdtype,
                                                       comm, module,
                                                       tuned_module->user_forced[ALLTOALLV].algorithm, &item);
#endif
    }

    /**
     * check to see if we have some filebased rules. As we don't have global
     * knowledge about the total amount of data, use the first available rule.
     * This allow the users to specify the alltoallv algorithm to be used only
     * based on the communicator size.
     */
    if (tuned_module->com_rules[ALLTOALLV]) {
        int alg, faninout, segsize, max_requests;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLTOALLV],
                                                        0, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_alltoallv_intra_do_this (sbuf, scounts, sdisps, sdtype,
                                                            rbuf, rcounts, rdisps, rdtype,
                                                            comm, module,
                                                            alg);
#else
            return ompi_coll_tuned_alltoallv_intra_do_this (sbuf, scounts, sdisps, sdtype,
                                                            rbuf, rcounts, rdisps, rdtype,
                                                            comm, module,
                                                            alg, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_alltoallv_intra_dec_fixed(sbuf, scounts, sdisps, sdtype,
                                                     rbuf, rcounts, rdisps, rdtype,
                                                     comm, module);
#else
    return ompi_coll_tuned_alltoallv_intra_dec_fixed(sbuf, scounts, sdisps, sdtype,
                                                     rbuf, rcounts, rdisps, rdtype,
                                                     comm, module, &item);
#endif
}

/*
 *    barrier_intra_dec
 *
 *    Function:    - seletects barrier algorithm to use
 *    Accepts:    - same arguments as MPI_Barrier()
 *    Returns:    - MPI_SUCCESS or error code (passed from the barrier implementation)
 */
int ompi_coll_tuned_barrier_intra_dec_dynamic(struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module
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
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,"ompi_coll_tuned_barrier_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[BARRIER].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_barrier_intra_do_this(comm, module,
                                                     tuned_module->user_forced[BARRIER].algorithm,
                                                     tuned_module->user_forced[BARRIER].tree_fanout,
                                                     tuned_module->user_forced[BARRIER].segsize);
#else
        return ompi_coll_tuned_barrier_intra_do_this(comm, module,
                                                     tuned_module->user_forced[BARRIER].algorithm,
                                                     tuned_module->user_forced[BARRIER].tree_fanout,
                                                     tuned_module->user_forced[BARRIER].segsize, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[BARRIER]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[BARRIER],
                                                        0, &faninout, &segsize, &ignoreme);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_barrier_intra_do_this (comm, module,
                                                          alg, faninout, segsize);
#else
            return ompi_coll_tuned_barrier_intra_do_this (comm, module,
                                                          alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */
#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_barrier_intra_dec_fixed (comm, module);
#else
    return ompi_coll_tuned_barrier_intra_dec_fixed (comm, module, &item);
#endif
}

/*
 *   bcast_intra_dec
 *
 *   Function:   - selects broadcast algorithm to use
 *   Accepts:   - same arguments as MPI_Bcast()
 *   Returns:   - MPI_SUCCESS or error code (passed from the bcast implementation)
 */
int ompi_coll_tuned_bcast_intra_dec_dynamic(void *buf, int count,
                                            struct ompi_datatype_t *dtype, int root,
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
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "coll:tuned:bcast_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[BCAST].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_bcast_intra_do_this(buf, count, dtype,
                                                   root, comm, module,
                                                   tuned_module->user_forced[BCAST].algorithm,
                                                   tuned_module->user_forced[BCAST].chain_fanout,
                                                   tuned_module->user_forced[BCAST].segsize);
#else
        if(item!=NULL){
            int algo = tuned_module->user_forced[BCAST].algorithm;
            if(algo == 1) memcpy(item->usedAlgorithm, "basic_linear", 12);
            else if(algo == 2) memcpy(item->usedAlgorithm, "chain", 5);
            else if(algo == 3) memcpy(item->usedAlgorithm, "pipeline", 8);
            else if(algo == 4) memcpy(item->usedAlgorithm, "split_binary_tree", 17);
            else if(algo == 5) memcpy(item->usedAlgorithm, "binary_tree", 11);
            else if(algo == 6) memcpy(item->usedAlgorithm, "binomial", 8);
            else if(algo == 7) memcpy(item->usedAlgorithm, "knomial", 7);
            else if(algo == 8) memcpy(item->usedAlgorithm, "scatter_allgather", 17);
            else if(algo == 9) memcpy(item->usedAlgorithm, "scatter_allgather_ring", 22);
        }
        return ompi_coll_tuned_bcast_intra_do_this(buf, count, dtype,
                                                   root, comm, module,
                                                   tuned_module->user_forced[BCAST].algorithm,
                                                   tuned_module->user_forced[BCAST].chain_fanout,
                                                   tuned_module->user_forced[BCAST].segsize, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[BCAST]) {
        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[BCAST],
                                                        dsize, &faninout, &segsize, &ignoreme);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_bcast_intra_do_this (buf, count, dtype, root,
                                                        comm, module,
                                                        alg, faninout, segsize);
#else
            if(item!=NULL){
                int algo = alg;
                if(algo == 1) memcpy(item->usedAlgorithm, "basic_linear", 12);
                else if(algo == 2) memcpy(item->usedAlgorithm, "chain", 5);
                else if(algo == 3) memcpy(item->usedAlgorithm, "pipeline", 8);
                else if(algo == 4) memcpy(item->usedAlgorithm, "split_binary_tree", 17);
                else if(algo == 5) memcpy(item->usedAlgorithm, "binary_tree", 11);
                else if(algo == 6) memcpy(item->usedAlgorithm, "binomial", 8);
                else if(algo == 7) memcpy(item->usedAlgorithm, "knomial", 7);
                else if(algo == 8) memcpy(item->usedAlgorithm, "scatter_allgather", 17);
                else if(algo == 9) memcpy(item->usedAlgorithm, "scatter_allgather_ring", 22);
            }
            return ompi_coll_tuned_bcast_intra_do_this (buf, count, dtype, root,
                                                        comm, module,
                                                        alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_bcast_intra_dec_fixed (buf, count, dtype, root,
                                                  comm, module);
#else
    return ompi_coll_tuned_bcast_intra_dec_fixed (buf, count, dtype, root,
                                                  comm, module, &item);
#endif
}

/*
 *    reduce_intra_dec
 *
 *    Function:    - seletects reduce algorithm to use
 *    Accepts:    - same arguments as MPI_reduce()
 *    Returns:    - MPI_SUCCESS or error code (passed from the reduce implementation)
 *
 */
int ompi_coll_tuned_reduce_intra_dec_dynamic( const void *sbuf, void *rbuf,
                                              int count, struct ompi_datatype_t* dtype,
                                              struct ompi_op_t* op, int root,
                                              struct ompi_communicator_t* comm,
                                              mca_coll_base_module_t *module
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

    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "coll:tuned:reduce_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[REDUCE].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_reduce_intra_do_this(sbuf, rbuf, count, dtype,
                                                    op, root, comm, module,
                                                    tuned_module->user_forced[REDUCE].algorithm,
                                                    tuned_module->user_forced[REDUCE].chain_fanout,
                                                    tuned_module->user_forced[REDUCE].segsize,
                                                    tuned_module->user_forced[REDUCE].max_requests);
#else
        return ompi_coll_tuned_reduce_intra_do_this(sbuf, rbuf, count, dtype,
                                                    op, root, comm, module,
                                                    tuned_module->user_forced[REDUCE].algorithm,
                                                    tuned_module->user_forced[REDUCE].chain_fanout,
                                                    tuned_module->user_forced[REDUCE].segsize,
                                                    tuned_module->user_forced[REDUCE].max_requests, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[REDUCE]) {

        /* we do, so calc the message size or what ever we need and use this for the evaluation */
        int alg, faninout, segsize, max_requests;
        size_t dsize;

        ompi_datatype_type_size(dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[REDUCE],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return  ompi_coll_tuned_reduce_intra_do_this (sbuf, rbuf, count, dtype,
                                                          op, root, comm, module,
                                                          alg, faninout,
                                                          segsize, max_requests);
#else
            return  ompi_coll_tuned_reduce_intra_do_this (sbuf, rbuf, count, dtype,
                                                          op, root, comm, module,
                                                          alg, faninout,
                                                          segsize, max_requests, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_reduce_intra_dec_fixed (sbuf, rbuf, count, dtype,
                                                   op, root, comm, module);
#else
    return ompi_coll_tuned_reduce_intra_dec_fixed (sbuf, rbuf, count, dtype,
                                                   op, root, comm, module, &item);
#endif
}

/*
 *    reduce_scatter_intra_dec
 *
 *    Function:   - seletects reduce_scatter algorithm to use
 *    Accepts:    - same arguments as MPI_Reduce_scatter()
 *    Returns:    - MPI_SUCCESS or error code (passed from
 *                  the reduce_scatter implementation)
 *
 */
int ompi_coll_tuned_reduce_scatter_intra_dec_dynamic(const void *sbuf, void *rbuf,
                                                     const int *rcounts,
                                                     struct ompi_datatype_t *dtype,
                                                     struct ompi_op_t *op,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif

    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "coll:tuned:reduce_scatter_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[REDUCESCATTER].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_reduce_scatter_intra_do_this(sbuf, rbuf, rcounts, dtype,
                                                            op, comm, module,
                                                            tuned_module->user_forced[REDUCESCATTER].algorithm,
                                                            tuned_module->user_forced[REDUCESCATTER].chain_fanout,
                                                            tuned_module->user_forced[REDUCESCATTER].segsize);
#else
        return ompi_coll_tuned_reduce_scatter_intra_do_this(sbuf, rbuf, rcounts, dtype,
                                                            op, comm, module,
                                                            tuned_module->user_forced[REDUCESCATTER].algorithm,
                                                            tuned_module->user_forced[REDUCESCATTER].chain_fanout,
                                                            tuned_module->user_forced[REDUCESCATTER].segsize, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[REDUCESCATTER]) {
        /* we do, so calc the message size or what ever we need and use
           this for the evaluation */
        int alg, faninout, segsize, ignoreme, i, count, size;
        size_t dsize;
        size = ompi_comm_size(comm);
        for (i = 0, count = 0; i < size; i++) { count += rcounts[i];}
        ompi_datatype_type_size (dtype, &dsize);
        dsize *= count;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[REDUCESCATTER],
                                                        dsize, &faninout,
                                                        &segsize, &ignoreme);
        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return  ompi_coll_tuned_reduce_scatter_intra_do_this (sbuf, rbuf, rcounts, dtype,
                                                                  op, comm, module,
                                                                  alg, faninout, segsize);
#else
            return  ompi_coll_tuned_reduce_scatter_intra_do_this (sbuf, rbuf, rcounts, dtype,
                                                                  op, comm, module,
                                                                  alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_reduce_scatter_intra_dec_fixed (sbuf, rbuf, rcounts,
                                                           dtype, op, comm, module);
#else
    return ompi_coll_tuned_reduce_scatter_intra_dec_fixed (sbuf, rbuf, rcounts,
                                                           dtype, op, comm, module, &item);
#endif
}

/*
 *    reduce_scatter_block_intra_dec
 *
 *    Function:   - seletects reduce_scatter_block algorithm to use
 *    Accepts:    - same arguments as MPI_Reduce_scatter_block()
 *    Returns:    - MPI_SUCCESS or error code (passed from
 *                  the reduce_scatter implementation)
 *
 */
int ompi_coll_tuned_reduce_scatter_block_intra_dec_dynamic(const void *sbuf, void *rbuf,
                                                           int rcount,
                                                           struct ompi_datatype_t *dtype,
                                                           struct ompi_op_t *op,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream, "coll:tuned:reduce_scatter_block_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[REDUCESCATTERBLOCK].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_reduce_scatter_block_intra_do_this(sbuf, rbuf, rcount, dtype,
                                                                  op, comm, module,
                                                                  tuned_module->user_forced[REDUCESCATTERBLOCK].algorithm,
                                                                  tuned_module->user_forced[REDUCESCATTERBLOCK].chain_fanout,
                                                                  tuned_module->user_forced[REDUCESCATTERBLOCK].segsize);
#else
        return ompi_coll_tuned_reduce_scatter_block_intra_do_this(sbuf, rbuf, rcount, dtype,
                                                                  op, comm, module,
                                                                  tuned_module->user_forced[REDUCESCATTERBLOCK].algorithm,
                                                                  tuned_module->user_forced[REDUCESCATTERBLOCK].chain_fanout,
                                                                  tuned_module->user_forced[REDUCESCATTERBLOCK].segsize, &item);
#endif
    }

    /* check to see if we have some filebased rules */
    if (tuned_module->com_rules[REDUCESCATTERBLOCK]) {
        /* we do, so calc the message size or what ever we need and use
           this for the evaluation */
        int alg, faninout, segsize, ignoreme, size;
        size_t dsize;
        size = ompi_comm_size(comm);
        ompi_datatype_type_size (dtype, &dsize);
        dsize *= rcount * size;

        alg = ompi_coll_tuned_get_target_method_params(tuned_module->com_rules[REDUCESCATTERBLOCK],
                                                       dsize, &faninout,
                                                       &segsize, &ignoreme);
        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return  ompi_coll_tuned_reduce_scatter_block_intra_do_this (sbuf, rbuf, rcount, dtype,
                                                                        op, comm, module,
                                                                        alg, faninout, segsize);
#else
            return  ompi_coll_tuned_reduce_scatter_block_intra_do_this (sbuf, rbuf, rcount, dtype,
                                                                        op, comm, module,
                                                                        alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /* end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_reduce_scatter_block_intra_dec_fixed (sbuf, rbuf, rcount,
                                                                 dtype, op, comm, module);
#else
    return ompi_coll_tuned_reduce_scatter_block_intra_dec_fixed (sbuf, rbuf, rcount,
                                                                 dtype, op, comm, module, &item);
#endif
}

/*
 *    allgather_intra_dec
 *
 *    Function:    - seletects allgather algorithm to use
 *    Accepts:    - same arguments as MPI_Allgather()
 *    Returns:    - MPI_SUCCESS or error code (passed from the selected
 *                        allgather function).
 */

int ompi_coll_tuned_allgather_intra_dec_dynamic(const void *sbuf, int scount,
                                                struct ompi_datatype_t *sdtype,
                                                void* rbuf, int rcount,
                                                struct ompi_datatype_t *rdtype,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_allgather_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLGATHER].algorithm) {
        /* User-forced algorithm */
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_allgather_intra_do_this(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       comm, module,
                                                       tuned_module->user_forced[ALLGATHER].algorithm,
                                                       tuned_module->user_forced[ALLGATHER].tree_fanout,
                                                       tuned_module->user_forced[ALLGATHER].segsize);
#else
        return ompi_coll_tuned_allgather_intra_do_this(sbuf, scount, sdtype,
                                                       rbuf, rcount, rdtype,
                                                       comm, module,
                                                       tuned_module->user_forced[ALLGATHER].algorithm,
                                                       tuned_module->user_forced[ALLGATHER].tree_fanout,
                                                       tuned_module->user_forced[ALLGATHER].segsize, NULL);
#endif
    }

    if (tuned_module->com_rules[ALLGATHER]) {
        /* We have file based rules:
           - calculate message size and other necessary information */
        int comsize;
        int alg, faninout, segsize, ignoreme;
        size_t dsize;

        ompi_datatype_type_size (sdtype, &dsize);
        comsize = ompi_comm_size(comm);
        dsize *= (ptrdiff_t)comsize * (ptrdiff_t)scount;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLGATHER],
                                                        dsize, &faninout, &segsize, &ignoreme);
        if (alg) {
            /* we have found a valid choice from the file based rules for
               this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_allgather_intra_do_this (sbuf, scount, sdtype,
                                                            rbuf, rcount, rdtype,
                                                            comm, module,
                                                            alg, faninout, segsize);
#else
            return ompi_coll_tuned_allgather_intra_do_this (sbuf, scount, sdtype,
                                                            rbuf, rcount, rdtype,
                                                            comm, module,
                                                            alg, faninout, segsize, &item);
#endif
        }
    }

    /* Use default decision */
#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_allgather_intra_dec_fixed (sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module);
#else
    return ompi_coll_tuned_allgather_intra_dec_fixed (sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module, &item);
#endif
}

/*
 *    allgatherv_intra_dec
 *
 *    Function:    - seletects allgatherv algorithm to use
 *    Accepts:    - same arguments as MPI_Allgatherv()
 *    Returns:    - MPI_SUCCESS or error code (passed from the selected
 *                        allgatherv function).
 */

int ompi_coll_tuned_allgatherv_intra_dec_dynamic(const void *sbuf, int scount,
                                                 struct ompi_datatype_t *sdtype,
                                                 void* rbuf, const int *rcounts,
                                                 const int *rdispls,
                                                 struct ompi_datatype_t *rdtype,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_allgatherv_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[ALLGATHERV].algorithm) {
        /* User-forced algorithm */
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_allgatherv_intra_do_this(sbuf, scount, sdtype,
                                                        rbuf, rcounts, rdispls, rdtype,
                                                        comm, module,
                                                        tuned_module->user_forced[ALLGATHERV].algorithm,
                                                        tuned_module->user_forced[ALLGATHERV].tree_fanout,
                                                        tuned_module->user_forced[ALLGATHERV].segsize);
#else
        return ompi_coll_tuned_allgatherv_intra_do_this(sbuf, scount, sdtype,
                                                        rbuf, rcounts, rdispls, rdtype,
                                                        comm, module,
                                                        tuned_module->user_forced[ALLGATHERV].algorithm,
                                                        tuned_module->user_forced[ALLGATHERV].tree_fanout,
                                                        tuned_module->user_forced[ALLGATHERV].segsize, &item);
#endif
    }

    if (tuned_module->com_rules[ALLGATHERV]) {
        /* We have file based rules:
           - calculate message size and other necessary information */
        int comsize, i;
        int alg, faninout, segsize, ignoreme;
        size_t dsize, total_size, per_rank_size;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (sdtype, &dsize);
        total_size = 0;
        for (i = 0; i < comsize; i++) { total_size += dsize * rcounts[i]; }

        per_rank_size = total_size / comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[ALLGATHERV],
                                                        per_rank_size, &faninout, &segsize, &ignoreme);
        if (alg) {
            /* we have found a valid choice from the file based rules for
               this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_allgatherv_intra_do_this (sbuf, scount, sdtype,
                                                             rbuf, rcounts,
                                                             rdispls, rdtype,
                                                             comm, module,
                                                             alg, faninout, segsize);
#else
            return ompi_coll_tuned_allgatherv_intra_do_this (sbuf, scount, sdtype,
                                                             rbuf, rcounts,
                                                             rdispls, rdtype,
                                                             comm, module,
                                                             alg, faninout, segsize, &item);
#endif
        }
    }
    /* Use default decision */
#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_allgatherv_intra_dec_fixed (sbuf, scount, sdtype,
                                                       rbuf, rcounts,
                                                       rdispls, rdtype,
                                                       comm, module);
#else
    return ompi_coll_tuned_allgatherv_intra_dec_fixed (sbuf, scount, sdtype,
                                                       rbuf, rcounts,
                                                       rdispls, rdtype,
                                                       comm, module, &item);
#endif
}

int ompi_coll_tuned_gather_intra_dec_dynamic(const void *sbuf, int scount,
                                             struct ompi_datatype_t *sdtype,
                                             void* rbuf, int rcount,
                                             struct ompi_datatype_t *rdtype,
                                             int root,
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

    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_gather_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[GATHER].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_gather_intra_do_this(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module,
                                                    tuned_module->user_forced[GATHER].algorithm,
                                                    tuned_module->user_forced[GATHER].tree_fanout,
                                                    tuned_module->user_forced[GATHER].segsize);
#else
        return ompi_coll_tuned_gather_intra_do_this(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module,
                                                    tuned_module->user_forced[GATHER].algorithm,
                                                    tuned_module->user_forced[GATHER].tree_fanout,
                                                    tuned_module->user_forced[GATHER].segsize, &item);
#endif
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[GATHER]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (sdtype, &dsize);
        dsize *= scount * comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[GATHER],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_gather_intra_do_this (sbuf, scount, sdtype,
                                                         rbuf, rcount, rdtype,
                                                         root, comm, module,
                                                         alg, faninout, segsize);
#else
            return ompi_coll_tuned_gather_intra_do_this (sbuf, scount, sdtype,
                                                         rbuf, rcount, rdtype,
                                                         root, comm, module,
                                                         alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */
#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_gather_intra_dec_fixed (sbuf, scount, sdtype,
                                                   rbuf, rcount, rdtype,
                                                   root, comm, module);
#else
    return ompi_coll_tuned_gather_intra_dec_fixed (sbuf, scount, sdtype,
                                                   rbuf, rcount, rdtype,
                                                   root, comm, module, &item);
#endif
}

int ompi_coll_tuned_scatter_intra_dec_dynamic(const void *sbuf, int scount,
                                              struct ompi_datatype_t *sdtype,
                                              void* rbuf, int rcount,
                                              struct ompi_datatype_t *rdtype,
                                              int root, struct ompi_communicator_t *comm,
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
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_scatter_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[SCATTER].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_scatter_intra_do_this(sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     root, comm, module,
                                                     tuned_module->user_forced[SCATTER].algorithm,
                                                     tuned_module->user_forced[SCATTER].chain_fanout,
                                                     tuned_module->user_forced[SCATTER].segsize);
#else
        return ompi_coll_tuned_scatter_intra_do_this(sbuf, scount, sdtype,
                                                     rbuf, rcount, rdtype,
                                                     root, comm, module,
                                                     tuned_module->user_forced[SCATTER].algorithm,
                                                     tuned_module->user_forced[SCATTER].chain_fanout,
                                                     tuned_module->user_forced[SCATTER].segsize, &item);
#endif
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[SCATTER]) {
      	  int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (sdtype, &dsize);
        dsize *= scount * comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[SCATTER],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_scatter_intra_do_this (sbuf, scount, sdtype,
                                                          rbuf, rcount, rdtype,
                                                          root, comm, module,
                                                          alg, faninout, segsize);
#else
            return ompi_coll_tuned_scatter_intra_do_this (sbuf, scount, sdtype,
                                                          rbuf, rcount, rdtype,
                                                          root, comm, module,
                                                          alg, faninout, segsize, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_tuned_scatter_intra_dec_fixed (sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module);
#else
    return ompi_coll_tuned_scatter_intra_dec_fixed (sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    root, comm, module, &item);
#endif
}

int ompi_coll_tuned_exscan_intra_dec_dynamic(const void *sbuf, void* rbuf, int count,
                                              struct ompi_datatype_t *dtype,
                                              struct ompi_op_t *op,
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
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_exscan_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[EXSCAN].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_exscan_intra_do_this(sbuf, rbuf, count, dtype,
                                                    op, comm, module,
                                                    tuned_module->user_forced[EXSCAN].algorithm);
#else
        return ompi_coll_tuned_exscan_intra_do_this(sbuf, rbuf, count, dtype,
                                                    op, comm, module,
                                                    tuned_module->user_forced[EXSCAN].algorithm, &item);
#endif
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[EXSCAN]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (dtype, &dsize);
        dsize *= comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[EXSCAN],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_exscan_intra_do_this (sbuf, rbuf, count, dtype,
                                                         op, comm, module,
                                                         alg);
#else
            return ompi_coll_tuned_exscan_intra_do_this (sbuf, rbuf, count, dtype,
                                                         op, comm, module,
                                                         alg, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */
#ifndef ENABLE_ANALYSIS
    return ompi_coll_base_exscan_intra_linear(sbuf, rbuf, count, dtype,
                                              op, comm, module);
#else
    return ompi_coll_base_exscan_intra_linear(sbuf, rbuf, count, dtype,
                                              op, comm, module, &item);
#endif
}

int ompi_coll_tuned_scan_intra_dec_dynamic(const void *sbuf, void* rbuf, int count,
                                           struct ompi_datatype_t *dtype,
                                           struct ompi_op_t *op,
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
        if(*q!=NULL){
            item = *q;
        } else item = NULL;
    } else item = NULL;
#endif
    mca_coll_tuned_module_t *tuned_module = (mca_coll_tuned_module_t*) module;

    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "ompi_coll_tuned_scan_intra_dec_dynamic"));

    /* Check first if an algorithm is set explicitly for this collective */
    if (tuned_module->user_forced[SCAN].algorithm) {
#ifndef ENABLE_ANALYSIS
        return ompi_coll_tuned_scan_intra_do_this(sbuf, rbuf, count, dtype,
                                                  op, comm, module,
                                                  tuned_module->user_forced[SCAN].algorithm);
#else
        return ompi_coll_tuned_scan_intra_do_this(sbuf, rbuf, count, dtype,
                                                  op, comm, module,
                                                  tuned_module->user_forced[SCAN].algorithm, &item);
#endif
    }

    /**
     * check to see if we have some filebased rules.
     */
    if (tuned_module->com_rules[SCAN]) {
        int comsize, alg, faninout, segsize, max_requests;
        size_t dsize;

        comsize = ompi_comm_size(comm);
        ompi_datatype_type_size (dtype, &dsize);
        dsize *= comsize;

        alg = ompi_coll_tuned_get_target_method_params (tuned_module->com_rules[SCAN],
                                                        dsize, &faninout, &segsize, &max_requests);

        if (alg) {
            /* we have found a valid choice from the file based rules for this message size */
#ifndef ENABLE_ANALYSIS
            return ompi_coll_tuned_scan_intra_do_this (sbuf, rbuf, count, dtype,
                                                       op, comm, module,
                                                       alg);
#else
            return ompi_coll_tuned_scan_intra_do_this (sbuf, rbuf, count, dtype,
                                                       op, comm, module,
                                                       alg, &item);
#endif
        } /* found a method */
    } /*end if any com rules to check */

#ifndef ENABLE_ANALYSIS
    return ompi_coll_base_scan_intra_linear(sbuf, rbuf, count, dtype,
                                            op, comm, module);
#else
    return ompi_coll_base_scan_intra_linear(sbuf, rbuf, count, dtype,
                                            op, comm, module, &item);
#endif
}
