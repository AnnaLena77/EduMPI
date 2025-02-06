/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015-2017 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2021      Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "nbc_internal.h"

#ifndef ENABLE_ANALYSIS
static inline int a2av_sched_linear(int rank, int p, NBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts,
                                    const int *sdispls, MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts,
                                    const int *rdispls, MPI_Aint rcvext, MPI_Datatype recvtype);

static inline int a2av_sched_pairwise(int rank, int p, NBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype);

static inline int a2av_sched_inplace(int rank, int p, NBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap, qentry **q);
#else
static inline int a2av_sched_linear(int rank, int p, NBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts,
                                    const int *sdispls, MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts,
                                    const int *rdispls, MPI_Aint rcvext, MPI_Datatype recvtype, qentry **q);

static inline int a2av_sched_pairwise(int rank, int p, NBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype, qentry **q);

static inline int a2av_sched_inplace(int rank, int p, NBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap, qentry **q);
#endif
/* an alltoallv schedule can not be cached easily because the contents
 * of the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* simple linear Alltoallv */
static int nbc_alltoallv_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                              MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                              MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              mca_coll_base_module_t *module, bool persistent
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
    } else item = NULL;
#endif
  int rank, p, res;
  MPI_Aint sndext, rcvext;
  NBC_Schedule *schedule;
  char *rbuf, *sbuf, inplace;
  ptrdiff_t gap = 0, span;
  void * tmpbuf = NULL;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_extent (recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  /* copy data to receivbuffer */
  if (inplace) {
    int count = 0;
    for (int i = 0; i < p; i++) {
      if (recvcounts[i] > count) {
        count = recvcounts[i];
      }
    }
    span = opal_datatype_span(&recvtype->super, count, &gap);
    /**
     * If this process has no data to send or receive it can bail out early,
     * but it needs to increase the nonblocking tag to stay in sync with the
     * rest of the processes.
     */
    if (OPAL_UNLIKELY(0 == span)) {
      ompi_coll_base_nbc_reserve_tags(comm, 1);
      return nbc_get_noop_request(persistent, request);
    }
    tmpbuf = malloc(span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    sendcounts = recvcounts;
    sdispls = rdispls;
    sndext = rcvext;
  } else {
    res = ompi_datatype_type_extent (sendtype, &sndext);
    if (MPI_SUCCESS != res) {
      NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
      return res;
    }
  }

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }


  if (!inplace && sendcounts[rank] != 0) {
    rbuf = (char *) recvbuf + rdispls[rank] * rcvext;
    sbuf = (char *) sendbuf + sdispls[rank] * sndext;
    res = NBC_Sched_copy (sbuf, false, sendcounts[rank], sendtype,
                          rbuf, false, recvcounts[rank], recvtype, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }
  }

  if (inplace) {
#ifdef ENABLE_ANALYSIS
    if(item != NULL) memcpy(item->usedAlgorithm, "sched_inplace", 13);
    res = a2av_sched_inplace(rank, p, schedule, recvbuf, recvcounts,
                                 rdispls, rcvext, recvtype, gap, &item);
#else
    res = a2av_sched_inplace(rank, p, schedule, recvbuf, recvcounts,
                                 rdispls, rcvext, recvtype, gap);
#endif
  } else {
#ifdef ENABLE_ANALYSIS
    if(item != NULL) memcpy(item->usedAlgorithm, "sched_linear", 12);
    res = a2av_sched_linear(rank, p, schedule,
                            sendbuf, sendcounts, sdispls, sndext, sendtype,
                            recvbuf, recvcounts, rdispls, rcvext, recvtype, &item);
#else
    res = a2av_sched_linear(rank, p, schedule,
                            sendbuf, sendcounts, sdispls, sndext, sendtype,
                            recvbuf, recvcounts, rdispls, rcvext, recvtype);
#endif
  }
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Sched_commit (schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_ialltoallv(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                mca_coll_base_module_t *module
#ifdef ENABLE_ANALYSIS
                                , qentry **q
#endif
                                ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
        int res = nbc_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype,
                                 recvbuf, recvcounts, rdispls, recvtype,
                                 comm, request, module, false, &item);
#else
    int res = nbc_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype,
                                 recvbuf, recvcounts, rdispls, recvtype,
                                 comm, request, module, false);
#endif
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

/* simple linear Alltoallv */
static int nbc_alltoallv_inter_init (const void* sendbuf, const int *sendcounts, const int *sdispls,
                                     MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                     MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     mca_coll_base_module_t *module, bool persistent
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
    } else item = NULL;
#endif
  int res, rsize;
  MPI_Aint sndext, rcvext;
  NBC_Schedule *schedule;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;


  res = ompi_datatype_type_extent(sendtype, &sndext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  rsize = ompi_comm_remote_size (comm);

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  for (int i = 0; i < rsize; i++) {
    /* post all sends */
    if (sendcounts[i] != 0) {
      char *sbuf = (char *) sendbuf + sdispls[i] * sndext;
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_send (sbuf, false, sendcounts[i], sendtype, i, schedule, false);
#else
      res = NBC_Sched_send (sbuf, false, sendcounts[i], sendtype, i, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
    /* post all receives */
    if (recvcounts[i] != 0) {
      char *rbuf = (char *) recvbuf + rdispls[i] * rcvext;
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_recv (rbuf, false, recvcounts[i], recvtype, i, schedule, false);
#else
      res = NBC_Sched_recv (rbuf, false, recvcounts[i], recvtype, i, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
  }

  res = NBC_Sched_commit(schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, NULL);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_ialltoallv_inter (const void* sendbuf, const int *sendcounts, const int *sdispls,
				       MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
				       MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
				       mca_coll_base_module_t *module
#ifdef ENABLE_ANALYSIS
                                           , qentry **q
#endif
				       ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
        int res = nbc_alltoallv_inter_init(sendbuf, sendcounts, sdispls, sendtype,
                                       recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request, module, false, &item);
#else
    int res = nbc_alltoallv_inter_init(sendbuf, sendcounts, sdispls, sendtype,
                                       recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request, module, false);
#endif
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

__opal_attribute_unused__
static inline int a2av_sched_linear(int rank, int p, NBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts, const int *sdispls,
                                    MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts, const int *rdispls,
                                    MPI_Aint rcvext, MPI_Datatype recvtype
#ifdef ENABLE_ANALYSIS
                                    , qentry **q
#endif
                                    ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
#endif
  int res;

  for (int i = 0 ; i < p ; ++i) {
    if (i == rank) {
      continue;
    }

    /* post send */
    if (sendcounts[i] != 0) {
      char *sbuf = ((char *) sendbuf) + (sdispls[i] * sndext);
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_send(sbuf, false, sendcounts[i], sendtype, i, schedule, false);
#else
      res = NBC_Sched_send(sbuf, false, sendcounts[i], sendtype, i, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* post receive */
    if (recvcounts[i] != 0) {
      char *rbuf = ((char *) recvbuf) + (rdispls[i] * rcvext);
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_recv(rbuf, false, recvcounts[i], recvtype, i, schedule, false);
#else
      res = NBC_Sched_recv(rbuf, false, recvcounts[i], recvtype, i, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

__opal_attribute_unused__
static inline int a2av_sched_pairwise(int rank, int p, NBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype
#ifdef ENABLE_ANALYSIS
                                          , qentry **q
#endif
                                      ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
#endif
  int res;

  for (int i = 1 ; i < p ; ++i) {
    int sndpeer = (rank + i) % p;
    int rcvpeer = (rank + p - i) %p;

    /* post send */
    if (sendcounts[sndpeer] != 0) {
      char *sbuf = ((char *) sendbuf) + (sdispls[sndpeer] * sndext);
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_send(sbuf, false, sendcounts[sndpeer], sendtype, sndpeer, schedule, false);
#else
      res = NBC_Sched_send(sbuf, false, sendcounts[sndpeer], sendtype, sndpeer, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* post receive */
    if (recvcounts[rcvpeer] != 0) {
      char *rbuf = ((char *) recvbuf) + (rdispls[rcvpeer] * rcvext);
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_recv(rbuf, false, recvcounts[rcvpeer], recvtype, rcvpeer, schedule, true);
#else
      res = NBC_Sched_recv(rbuf, false, recvcounts[rcvpeer], recvtype, rcvpeer, schedule, true, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2av_sched_inplace(int rank, int p, NBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap
#ifdef ENABLE_ANALYSIS
                                    , qentry **q
#endif
                                    ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
#endif
  int res;

  for (int i = 1; i < (p+1)/2; i++) {
    int speer = (rank + i) % p;
    int rpeer = (rank + p - i) % p;
    char *sbuf = (char *) buf + displs[speer] * ext;
    char *rbuf = (char *) buf + displs[rpeer] * ext;

    if (0 != counts[rpeer]) {
      res = NBC_Sched_copy (rbuf, false, counts[rpeer], type,
                            (void *)(-gap), true, counts[rpeer], type,
                            schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[speer]) {
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_send (sbuf, false , counts[speer], type, speer, schedule, false);
#else
      res = NBC_Sched_send (sbuf, false , counts[speer], type, speer, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[rpeer]) {
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_recv (rbuf, false , counts[rpeer], type, rpeer, schedule, true);
#else
      res = NBC_Sched_recv (rbuf, false , counts[rpeer], type, rpeer, schedule, true, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    if (0 != counts[rpeer]) {
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_send ((void *)(-gap), true, counts[rpeer], type, rpeer, schedule, false);
#else
      res = NBC_Sched_send ((void *)(-gap), true, counts[rpeer], type, rpeer, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[speer]) {
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_recv (sbuf, false, counts[speer], type, speer, schedule, true);
#else
      res = NBC_Sched_recv (sbuf, false, counts[speer], type, speer, schedule, true, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }
  if (0 == (p%2)) {
    int peer = (rank + p/2) % p;

    char *tbuf = (char *) buf + displs[peer] * ext;
    res = NBC_Sched_copy (tbuf, false, counts[peer], type,
                          (void *)(-gap), true, counts[peer], type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    if (0 != counts[peer]) {
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_send ((void *)(-gap), true , counts[peer], type, peer, schedule, false);
#else
      res = NBC_Sched_send ((void *)(-gap), true , counts[peer], type, peer, schedule, false, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
#ifndef ENABLE_ANALYSIS
      res = NBC_Sched_recv (tbuf, false , counts[peer], type, peer, schedule, true);
#else
      res = NBC_Sched_recv (tbuf, false , counts[peer], type, peer, schedule, true, &item);
#endif
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_alltoallv_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                    MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    mca_coll_base_module_t *module
#ifdef ENABLE_ANALYSIS
                                    , qentry **q
#endif
                                    ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
        int res = nbc_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                 comm, request, module, true, &item);
#else
    int res = nbc_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                 comm, request, module, true);
#endif
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_alltoallv_inter_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                          MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          mca_coll_base_module_t *module
#ifdef ENABLE_ANALYSIS
                                          , qentry **q
#endif
                                          ) {
#ifdef ENABLE_ANALYSIS
    qentry *item;
    if(q!=NULL){
        if(*q!=NULL) {
            item = *q;
        }
        else item = NULL;
    } else item = NULL;
        int res = nbc_alltoallv_inter_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request, module, true, &item);
#else
    int res = nbc_alltoallv_inter_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request, module, true);
#endif
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
