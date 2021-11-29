/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2006 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2018 Cisco Systems, Inc.  All rights reserved
 * Copyright (c) 2007-2008 Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <sys/queue.h>
#include <mysql/mysql.h>

#include "opal/util/show_help.h"
#include "ompi/runtime/ompi_spc.h"
#include "ompi/mpi/c/bindings.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/constants.h"
#include "ompi/mpi/c/init.h"


#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Init = PMPI_Init
#endif
#define MPI_Init PMPI_Init
#endif

typedef struct qentry {
    char* type;
    int data;
    TAILQ_ENTRY(qentry) pointers;
} qentry;

TAILQ_HEAD(, qentry) head;

void enqueue(char* type, int value){
    qentry *item = (qentry*)malloc(sizeof(qentry));
    item->data = value;
    item->type = type;
    TAILQ_INSERT_TAIL(&head, item, pointers);
}

qentry* dequeue(){
    qentry *item;
    item = TAILQ_FIRST(&head);
    TAILQ_REMOVE(&head, item, pointers);
    return item;
}

MYSQL *conn;
MYSQL_RES *res;
MYSQL_ROW row;
char *server = "192.168.42.9";
char *user = "AnnaLena";
char *password = "annalena";
char *database = "DataFromMPI";

static void insertData(qentry **item){
    qentry *q = *item;
    char query[300];
    sprintf(query, "INSERT INTO MPI_Data(type, value)VALUES('%s', %d)", q->type, q->data);
    //printf("%s\n", query);
    if(mysql_query(conn, query)){
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
}

pthread_t MONITOR_THREAD;

 
static void* MonitorFunc(void* _arg){
    qentry *item;
    int finish = 0;
    while(!finish){
        if(TAILQ_EMPTY(&head)){
            sleep(2);
            if(TAILQ_EMPTY(&head)){
                finish = 1; 
            }
        }
        else {
            item = dequeue();
            insertData(&item);
            //printf("%d\n", item->data);
        }
    }
}

static const char FUNC_NAME[] = "MPI_Init";


int MPI_Init(int *argc, char ***argv)
{   
    conn = mysql_init(NULL);
    if(!mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)){
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
    else {
    	if(mysql_query(conn, "DELETE FROM MPI_Data")){
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    	}
    }
    
    TAILQ_INIT(&head);
    pthread_create(&MONITOR_THREAD, NULL, MonitorFunc, NULL);
    int err;
    int provided;
    char *env;
    int required = MPI_THREAD_SINGLE;

    /* check for environment overrides for required thread level.  If
       there is, check to see that it is a valid/supported thread level.
       If not, default to MPI_THREAD_MULTIPLE. */

    if (NULL != (env = getenv("OMPI_MPI_THREAD_LEVEL"))) {
        required = atoi(env);
        if (required < MPI_THREAD_SINGLE || required > MPI_THREAD_MULTIPLE) {
            required = MPI_THREAD_MULTIPLE;
        }
    }

    /* Call the back-end initialization function (we need to put as
       little in this function as possible so that if it's profiled, we
       don't lose anything) */

    if (NULL != argc && NULL != argv) {
        err = ompi_mpi_init(*argc, *argv, required, &provided, false);
    } else {
        err = ompi_mpi_init(0, NULL, required, &provided, false);
    }

    /* Since we don't have a communicator to invoke an errorhandler on
       here, don't use the fancy-schmancy ERRHANDLER macros; they're
       really designed for real communicator objects.  Just use the
       back-end function directly. */

    if (MPI_SUCCESS != err) {
        return ompi_errhandler_invoke(NULL, NULL,
                                      OMPI_ERRHANDLER_TYPE_COMM,
                                      err <
                                      0 ? ompi_errcode_get_mpi_code(err) :
                                      err, FUNC_NAME);
    }

    SPC_INIT();

    return MPI_SUCCESS;
}
