/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2016 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2014-2021 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
 
 
 /* Info zu OPAL_ERROR_CODES = MPI_ERROR_CODES
  * 
  * OPAL_SUCCESS = OPAL_ERR_BASE (initialisiert mit 0)
  * OPAL_ERROR = OPAL_ERR_BASE-1
  * OPAL_ERR_NOT_AVAILABLE = OPAL_ERR_BASE-16
  * OPAL_ERR_UNREACH = OPAL_ERR_BASE-12
 */

#include "ompi_config.h"

#include "pml_ob1.h"
#include "pml_ob1_sendreq.h"
#include "pml_ob1_recvreq.h"
#include "ompi/peruse/peruse-internal.h"
#include "ompi/runtime/ompi_spc.h"
#if MPI_VERSION >= 4
#include "ompi/mca/pml/base/pml_base_sendreq.h"
#endif


#ifdef ENABLE_ANALYSIS
struct timespec wait_start;
struct timespec wait_end;
#endif

/*#ifdef ENABLE_ANALYSIS
#include "ompi/mpi/c/init.h"
#endif*/


/**
 * Single usage request. As we allow recursive calls (as an
 * example from the request completion callback), we cannot rely
 * on using a global request. Thus, once a send acquires ownership
 * of this global request, it should set it to NULL to prevent
 * the reuse until the first user completes.
 */
mca_pml_ob1_send_request_t *mca_pml_ob1_sendreq = NULL;

int mca_pml_ob1_isend_init(const void *buf,
                           size_t count,
                           ompi_datatype_t * datatype,
                           int dst,
                           int tag,
                           mca_pml_base_send_mode_t sendmode,
                           ompi_communicator_t * comm,
                           ompi_request_t ** request
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
    } else {
        item = NULL;
    }
#endif
    mca_pml_ob1_comm_proc_t *ob1_proc = mca_pml_ob1_peer_lookup (comm, dst);
    mca_pml_ob1_send_request_t *sendreq = NULL;
    MCA_PML_OB1_SEND_REQUEST_ALLOC(comm, dst, sendreq);
    if (NULL == sendreq)
        return OMPI_ERR_OUT_OF_RESOURCE;

    MCA_PML_OB1_SEND_REQUEST_INIT(sendreq, buf, count, datatype, dst, tag,
                                  comm, sendmode, true, ob1_proc);

#ifdef ENABLE_ANALYSIS
    if(item!=NULL){ 
        sendreq->q = item;
        clock_gettime(CLOCK_REALTIME, &sendreq->activate);
    }
#endif
    PERUSE_TRACE_COMM_EVENT (PERUSE_COMM_REQ_ACTIVATE,
                             &(sendreq)->req_send.req_base,
                             PERUSE_SEND);

    /* Work around a leak in start by marking this request as complete. The
     * problem occurred because we do not have a way to differentiate an
     * initial request and an incomplete pml request in start. This line
     * allows us to detect this state. */
    sendreq->req_send.req_base.req_pml_complete = true;

    *request = (ompi_request_t *) sendreq;
    return OMPI_SUCCESS;
}

/* try to get a small message out on to the wire quickly */
#ifndef ENABLE_ANALYSIS
static inline int mca_pml_ob1_send_inline (const void *buf, size_t count,
                                           ompi_datatype_t * datatype,
                                           int dst, int tag, int16_t seqn,
                                           ompi_proc_t *dst_proc, mca_pml_ob1_comm_proc_t *ob1_proc,
                                           mca_bml_base_endpoint_t* endpoint,
                                           ompi_communicator_t * comm)
{
#else
static inline int mca_pml_ob1_send_inline (const void *buf, size_t count,
                                           ompi_datatype_t * datatype,
                                           int dst, int tag, int16_t seqn,
                                           ompi_proc_t *dst_proc, mca_pml_ob1_comm_proc_t *ob1_proc, mca_bml_base_endpoint_t* endpoint,
                                           ompi_communicator_t * comm, qentry **q)
{
#endif
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
    mca_pml_ob1_match_hdr_t match;
    mca_bml_base_btl_t *bml_btl;
    opal_convertor_t convertor;
    size_t size;
    int rc;

    //Error, wenn kein BTL verfügbar
    bml_btl = mca_bml_base_btl_array_get_next(&endpoint->btl_eager); //btl->eager ist das array der btls to use for first
    if( NULL == bml_btl || NULL == bml_btl->btl->btl_sendi)
        return OMPI_ERR_NOT_AVAILABLE; //Zahl: -16
        
    //Error, wenn Größe des Datentyps * Anzahl >256 --> Kein Eagersend, sondern Rendevouz
    ompi_datatype_type_size (datatype, &size);

    /* the size used here was picked based on performance on a Cray XE-6. it should probably
     * be provided by the btl module */
    if ((size * count) > 256 || -1 == ob1_proc->comm_index) {
        return OMPI_ERR_NOT_AVAILABLE;
    }
#ifdef ENABLE_ANALYSIS
    else {
        if(item != NULL) item->withinEagerLimit = 1;
    }
#endif

    if (count > 0) {
        /* initialize just enough of the convertor to avoid a SEGV in opal_convertor_cleanup */
        OBJ_CONSTRUCT(&convertor, opal_convertor_t);

        /* We will create a convertor specialized for the        */
        /* remote architecture and prepared with the datatype.   */
        opal_convertor_copy_and_prepare_for_send (dst_proc->super.proc_convertor,
                                                  (const struct opal_datatype_t *) datatype,
                                                  count, buf, 0, &convertor);
        opal_convertor_get_packed_size (&convertor, &size);
    } else {
        size = 0;
    }
    
    //Header-Match vorbereiten
    mca_pml_ob1_match_hdr_prepare (&match, MCA_PML_OB1_HDR_TYPE_MATCH, 0,
                                   ob1_proc->comm_index, comm->c_my_rank,
                                   tag, seqn);

    ob1_hdr_hton(&match, MCA_PML_OB1_HDR_TYPE_MATCH, dst_proc);

    /* try to send immediately, defined in bml.h, return btl->btl_sendi */
#ifndef ENABLE_ANALYSIS
    rc = mca_bml_base_sendi (bml_btl, &convertor, &match, OMPI_PML_OB1_MATCH_HDR_LEN,
                             size, MCA_BTL_NO_ORDER, MCA_BTL_DES_FLAGS_PRIORITY | MCA_BTL_DES_FLAGS_BTL_OWNERSHIP,
                             MCA_PML_OB1_HDR_TYPE_MATCH, NULL);
#else
    //btl_sendi Callback in btl.h, Definitionen in verschiedenen btls: self, uct, sm, ugni, smcuda
    rc = mca_bml_base_sendi (bml_btl, &convertor, &match, OMPI_PML_OB1_MATCH_HDR_LEN,
                             size, MCA_BTL_NO_ORDER, MCA_BTL_DES_FLAGS_PRIORITY | MCA_BTL_DES_FLAGS_BTL_OWNERSHIP,
                             MCA_PML_OB1_HDR_TYPE_MATCH, NULL, &item);
#endif

    /* This #if is required due to an issue that arises with the IBM CI (XL Compiler).
     * The compiler doesn't seem to like having a compiler hint attached to an if
     * statement that only has a no-op inside and has the following error:
     *
     * 1586-494 (U) INTERNAL COMPILER ERROR: Signal 11.
     */
#if SPC_ENABLE == 1
    if(OPAL_LIKELY(rc == OPAL_SUCCESS)) {
        SPC_USER_OR_MPI(tag, (ompi_spc_value_t)size, OMPI_SPC_BYTES_SENT_USER, OMPI_SPC_BYTES_SENT_MPI);
    }
#endif

    if (count > 0) {
        opal_convertor_cleanup (&convertor);
    }

    if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
	return rc;
    }
    return (int) size;
}

#ifndef ENABLE_ANALYSIS
int mca_pml_ob1_isend(const void *buf,
                      size_t count,
                      ompi_datatype_t * datatype,
                      int dst,
                      int tag,
                      mca_pml_base_send_mode_t sendmode,
                      ompi_communicator_t * comm,
                      ompi_request_t ** request)
{
#else
int mca_pml_ob1_isend(const void *buf,
                      size_t count,
                      ompi_datatype_t * datatype,
                      int dst,
                      int tag,
                      mca_pml_base_send_mode_t sendmode,
                      ompi_communicator_t * comm,
                      ompi_request_t ** request, qentry ** q)
{
#endif
#ifdef ENABLE_ANALYSIS
    //printf("Funktion: Ob1_isend\n");
    qentry *item;
    //if q is NULL, isend is not called from a normal operation
    if(q!=NULL){
        if(*q!=NULL){
            item = *q;
            //printf("Isend: %s\n", item->function);
            item->sendcount = item->sendcount + count;
        	   if(datatype == MPI_PACKED){
                item->sendDatasize += count;
            } else{
                int type_size = 0;
                MPI_Type_size(datatype, &type_size);
        	       item->sendDatasize += count*type_size;
            }
            if(item->blocking == 0 && !strcmp(item->communicationType, "p2p")){
                //qentry->sendmode & qentry->operation
                if(sendmode==MCA_PML_BASE_SEND_SYNCHRONOUS){
                    strcpy(item->sendmode, "SYNCHRONOUS");
                }
                else if(sendmode==MCA_PML_BASE_SEND_BUFFERED){
                    strcpy(item->sendmode, "BUFFERED");
                }
                else if(sendmode==MCA_PML_BASE_SEND_READY){
                    strcpy(item->sendmode, "READY");
                }
                else if(sendmode==MCA_PML_BASE_SEND_STANDARD){
                    strcpy(item->sendmode, "STANDARD");
                }
            } else if(!strcmp(item->communicationType, "collective")){
                item->coll_partnerranks[dst / 8] |= (1 << (dst % 8));
            }
        }else item = NULL;
    }
    else {
        item = NULL;
    }
#endif
    mca_pml_ob1_comm_proc_t *ob1_proc = mca_pml_ob1_peer_lookup (comm, dst);
    mca_pml_ob1_send_request_t *sendreq = NULL;
    ompi_proc_t *dst_proc = ob1_proc->ompi_proc;
    mca_bml_base_endpoint_t* endpoint = mca_bml_base_get_endpoint (dst_proc);
    int16_t seqn = 0;
    int rc;

    if (OPAL_UNLIKELY(NULL == endpoint)) {
#if OPAL_ENABLE_FT_MPI
        if (!dst_proc->proc_active) {
            goto alloc_ft_req;
        }
#endif /* OPAL_ENABLE_FT_MPI */
        return OMPI_ERR_UNREACH;
    }

    if (!OMPI_COMM_CHECK_ASSERT_ALLOW_OVERTAKE(comm) || 0 > tag) {
        seqn = (uint16_t) OPAL_THREAD_ADD_FETCH32(&ob1_proc->send_sequence, 1);
    }

    if (MCA_PML_BASE_SEND_SYNCHRONOUS != sendmode) {

#ifndef ENABLE_ANALYSIS
        rc = mca_pml_ob1_send_inline (buf, count, datatype, dst, tag, seqn, dst_proc, ob1_proc,
                                      endpoint, comm);
#else
        rc = mca_pml_ob1_send_inline (buf, count, datatype, dst, tag, seqn, dst_proc, ob1_proc,
                                      endpoint, comm, &item);
#endif
        //Wenn rc>0 -> Error aus send_inline!
        //Ansonsten ist rc die size, die durch opal_convertor_get_packed_size berechnet wird.
        //Wenn rc = 0, keine Daten versendet.
        if (OPAL_LIKELY(0 <= rc)) {
            /* NTH: it is legal to return ompi_request_empty since the only valid
             * field in a send completion status is whether or not the send was
             * cancelled (which it can't be at this point anyway). */
#if MPI_VERSION >= 4
            if (OPAL_UNLIKELY(OMPI_PML_BASE_WARN_DEP_CANCEL_SEND_NEVER != ompi_pml_base_warn_dep_cancel_send_level)) {
                *request = &ompi_request_empty_send;
                return OMPI_SUCCESS;
            }
#endif
            *request = &ompi_request_empty;
            return OMPI_SUCCESS;
        }
    }
    //siehe pml_ob1_sendreq.h
    //Speicherplatz für den Request allocieren
    MCA_PML_OB1_SEND_REQUEST_ALLOC(comm, dst, sendreq);
    if (NULL == sendreq){
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    //siehe pml_base_sendreq.h
    //MCA_PML_BASE_SEND_REQUEST_INIT
    //Funktionsaufrufe: opal_converter_copy_and_prepare_for_send, opal_converter_get_packed_size
    MCA_PML_OB1_SEND_REQUEST_INIT(sendreq,
                                  buf,
                                  count,
                                  datatype,
                                  dst, tag,
                                  comm, sendmode, false, ob1_proc);

    /*#ifdef ENABLE_ANALYSIS
    if(item!=NULL) clock_gettime(CLOCK_REALTIME, &item->initializeRequest);
    #endif*/
#ifdef ENABLE_ANALYSIS
    if(item!=NULL){ 
        sendreq->q = item;
        clock_gettime(CLOCK_REALTIME, &sendreq->activate);
        //printf("Isend: %s\n", item->function);
    }
#endif
    
    PERUSE_TRACE_COMM_EVENT (PERUSE_COMM_REQ_ACTIVATE,
                             &(sendreq)->req_send.req_base,
                             PERUSE_SEND);
    
    MCA_PML_OB1_SEND_REQUEST_START_W_SEQ(sendreq, endpoint, seqn, rc);
    *request = (ompi_request_t *) sendreq;
#ifdef ENABLE_ANALYSIS
    //if(item!=NULL) qentryIntoQueue(&item);
#endif
    return rc;

#if OPAL_ENABLE_FT_MPI
//printf("OPAL_ENABLE_FT_MPI\n");
alloc_ft_req:
    MCA_PML_OB1_SEND_REQUEST_ALLOC(comm, dst, sendreq);
    if (NULL == sendreq)
        return OMPI_ERR_OUT_OF_RESOURCE;

    MCA_PML_OB1_SEND_REQUEST_INIT(sendreq,
                                  buf,
                                  count,
                                  datatype,
                                  dst, tag,
                                  comm, sendmode, false, ob1_proc);

#ifdef ENABLE_ANALYSIS
    if(item!=NULL){ 
        sendreq->q = item;
        clock_gettime(CLOCK_REALTIME, &sendreq->activate);
        //printf("Isend: %s\n", item->function);
    }
#endif

    PERUSE_TRACE_COMM_EVENT (PERUSE_COMM_REQ_ACTIVATE,
                             &(sendreq)->req_send.req_base,
                             PERUSE_SEND);

    /* No point in starting the request, it won't go through, mark completed
     * in error for collection in future wait */
    sendreq->req_send.req_base.req_ompi.req_status.MPI_ERROR = ompi_comm_is_revoked(comm)? MPI_ERR_REVOKED: MPI_ERR_PROC_FAILED;
    MCA_PML_OB1_SEND_REQUEST_MPI_COMPLETE(sendreq, false);
    OPAL_OUTPUT_VERBOSE((2, ompi_ftmpi_output_handle, "Allocating request in error %p (peer %d, seq %" PRIu64 ") with error code %d",
                         (void*) sendreq,
                         dst,
                         sendreq->req_send.req_base.req_sequence,
                         sendreq->req_send.req_base.req_ompi.req_status.MPI_ERROR));
    *request = (ompi_request_t *) sendreq;
    return OMPI_SUCCESS;
#endif /* OPAL_ENABLE_FT_MPI */
}

#ifndef ENABLE_ANALYSIS
int mca_pml_ob1_send(const void *buf,
                     size_t count,
                     ompi_datatype_t * datatype,
                     int dst,
                     int tag,
                     mca_pml_base_send_mode_t sendmode,
                     ompi_communicator_t * comm)
{
#else
int mca_pml_ob1_send(const void *buf,
                     size_t count,
                     ompi_datatype_t * datatype,
                     int dst,
                     int tag,
                     mca_pml_base_send_mode_t sendmode,
                     ompi_communicator_t * comm, qentry **q)
{
#endif
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL){
            item = *q;
            //printf("Send: %s\n", item->function);
        }else item = NULL;
    }
    else {
        item = NULL;
    }
#endif
    //printf("Funktion: Ob1_send\n");
    mca_pml_ob1_comm_proc_t *ob1_proc = mca_pml_ob1_peer_lookup (comm, dst);
    ompi_proc_t *dst_proc = ob1_proc->ompi_proc;
    
#ifdef ENABLE_ANALYSIS
    opal_proc_t *dst_opal_proc = &dst_proc->super;
    //opal_proc_t_name *dst_name = dst_opal_proc->opal_process_name_t;
    opal_vpid_t dst_vpid = dst_opal_proc->proc_name.vpid;
    printf("Globaler Rank (vpid) in MPI_COMM_WORLD: %" PRIu32 "\n", dst_vpid);
   // printf("Globaler Rank (vpid) in MPI_COMM_WORLD: %ls\n", dst_vpid);
#endif
    
    mca_bml_base_endpoint_t* endpoint = mca_bml_base_get_endpoint (dst_proc);
    mca_pml_ob1_send_request_t *sendreq = NULL;
    int16_t seqn = 0;
    int rc;

    if (OPAL_UNLIKELY(NULL == endpoint)) {
#if OPAL_ENABLE_FT_MPI
        if (!dst_proc->proc_active) {
            return MPI_ERR_PROC_FAILED;
        }
#endif /* OPAL_ENABLE_FT_MPI */
        return OMPI_ERR_UNREACH;
    }

    if (OPAL_UNLIKELY(MCA_PML_BASE_SEND_BUFFERED == sendmode)) {
        /* large buffered sends *need* a real request so use isend instead */
        ompi_request_t *brequest;
#ifndef ENABLE_ANALYSIS
        rc = mca_pml_ob1_isend (buf, count, datatype, dst, tag, sendmode, comm, &brequest);
#else
        strcpy(item->sendmode, "BUFFERED");
        rc = mca_pml_ob1_isend (buf, count, datatype, dst, tag, sendmode, comm, &brequest, &item);
#endif
        if (OPAL_UNLIKELY(OMPI_SUCCESS != rc)) {
            return rc;
        }
        //Wartet auf Receive, der gepostet sein muss, um Deadlock zu vermeiden
        //MPI_Bsend garantiert, dass Data in Buffer kopiert wurde, aber nicht, dass die Nachricht kopiert wurde 
        //Buffer kann sicher wieder verwendet werden
#ifdef ENABLE_ANALYSIS
    if(item!=NULL) clock_gettime(CLOCK_MONOTONIC, &wait_start);
#endif
        ompi_request_wait_completion (brequest);
#ifdef ENABLE_ANALYSIS
        if(item!=NULL){
            clock_gettime(CLOCK_MONOTONIC, &wait_end);
            long long elapsed_ns = (wait_end.tv_sec - wait_start.tv_sec) * 1000000000LL + (wait_end.tv_nsec - wait_start.tv_nsec);
            item->sendWaitingTime += (double)elapsed_ns / 1e9;
        }    
#endif
        ompi_request_free (&brequest);
        return OMPI_SUCCESS;
    }
#ifdef ENABLE_ANALYSIS
    //else {
        if(item!=NULL){
            //size of send-Data:
            item->sendcount = item->sendcount + count;
            if(datatype == MPI_PACKED){
                item->sendDatasize += count;
            } else{
        	       int type_size = 0;
                MPI_Type_size(datatype, &type_size);
        	       item->sendDatasize += count*type_size;
            }
            if(!strcmp(item->communicationType, "p2p")){
                if(sendmode==MCA_PML_BASE_SEND_SYNCHRONOUS){
                    strcpy(item->sendmode, "SYNCHRONOUS");
                }
                else if(sendmode==MCA_PML_BASE_SEND_READY){
                    strcpy(item->sendmode, "READY");
                }
                else if(sendmode==MCA_PML_BASE_SEND_STANDARD){
                    strcpy(item->sendmode, "STANDARD");
                }
            }
            else if(!strcmp(item->communicationType, "collective")){
                item->coll_partnerranks[dst / 8] |= (1 << (dst % 8));
            }
        }
    //}
#endif

    if (!OMPI_COMM_CHECK_ASSERT_ALLOW_OVERTAKE(comm) || 0 > tag) {
        seqn = (uint16_t) OPAL_THREAD_ADD_FETCH32(&ob1_proc->send_sequence, 1);
    }

    /**
     * The immediate send will not have a request, so they are
     * intracable from the point of view of any debugger attached to
     * the parallel application.
     */
    if (MCA_PML_BASE_SEND_SYNCHRONOUS != sendmode) {
#ifndef ENABLE_ANALYSIS
        rc = mca_pml_ob1_send_inline (buf, count, datatype, dst, tag, seqn, dst_proc,
                                      ob1_proc, endpoint, comm);
#else
        rc = mca_pml_ob1_send_inline (buf, count, datatype, dst, tag, seqn, dst_proc,
                                      ob1_proc, endpoint, comm, &item);
#endif
        //Sowohl MPI_Send als auch MPI_Rsend!
        //Wenn rc<0 -> Error aus send_inline!
        //Ansonsten ist rc die size, die durch opal_convertor_get_packed_size berechnet wird.
        //Wenn rc = 0, keine Daten versendet.

        if (OPAL_LIKELY(0 <= rc)) {
            return OMPI_SUCCESS;
        }
        //Wenn MPI_Send Nachrichtengröße >16 KB, Rendevous-Protokoll! 0>rc!
    }

    if (OPAL_LIKELY(!ompi_mpi_thread_multiple)) {
        sendreq = mca_pml_ob1_sendreq;
        mca_pml_ob1_sendreq = NULL;
    }

    if( OPAL_UNLIKELY(NULL == sendreq) ) {
        MCA_PML_OB1_SEND_REQUEST_ALLOC(comm, dst, sendreq);
        if (NULL == sendreq)
            return OMPI_ERR_TEMP_OUT_OF_RESOURCE;
    }
    sendreq->req_send.req_base.req_proc = dst_proc;
    sendreq->rdma_frag = NULL;

    MCA_PML_OB1_SEND_REQUEST_INIT(sendreq, buf, count, datatype, dst, tag,
                                  comm, sendmode, false, ob1_proc);
                                  
#ifdef ENABLE_ANALYSIS
    if(item!=NULL){ 
        sendreq->q = item;
        clock_gettime(CLOCK_REALTIME, &sendreq->activate);
        //printf("Isend: %s\n", item->function);
    }
#endif
    
    PERUSE_TRACE_COMM_EVENT (PERUSE_COMM_REQ_ACTIVATE,
                             &sendreq->req_send.req_base,
                             PERUSE_SEND);
    MCA_PML_OB1_SEND_REQUEST_START_W_SEQ(sendreq, endpoint, seqn, rc);
    if (OPAL_LIKELY(rc == OMPI_SUCCESS)) {
/*#ifdef ENABLE_ANALYSIS
        if(item!=NULL) clock_gettime(CLOCK_REALTIME, &item->requestWaitCompletion);
#endif*/
#ifdef ENABLE_ANALYSIS
    if(item!=NULL) clock_gettime(CLOCK_MONOTONIC, &wait_start);
#endif
        ompi_request_wait_completion(&sendreq->req_send.req_base.req_ompi);
#ifdef ENABLE_ANALYSIS
    if(item!=NULL){
        clock_gettime(CLOCK_MONOTONIC, &wait_end);
        long long elapsed_ns = (wait_end.tv_sec - wait_start.tv_sec) * 1000000000LL + (wait_end.tv_nsec - wait_start.tv_nsec);
        item->sendWaitingTime += (double)elapsed_ns / 1e9;
    } 
#endif

        rc = sendreq->req_send.req_base.req_ompi.req_status.MPI_ERROR;
    }

    if (OPAL_UNLIKELY(ompi_mpi_thread_multiple || NULL != mca_pml_ob1_sendreq)) {
        MCA_PML_OB1_SEND_REQUEST_RETURN(sendreq);
    } else {
        mca_pml_ob1_send_request_fini (sendreq);
        mca_pml_ob1_sendreq = sendreq;
    }
/*#ifdef ENABLE_ANALYSIS
    if(item!=NULL){ 
        clock_gettime(CLOCK_REALTIME, &item->requestFini);
        //qentryIntoQueue(&item);
    }
#endif*/
    return rc;
}
