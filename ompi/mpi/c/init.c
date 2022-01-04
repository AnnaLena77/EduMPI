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
#include <math.h>
#include <sys/queue.h>
#include <sys/time.h>
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
    char* operation;
    char* datatype;
    int count;
    int datasize;
    char* communicator;
    int processrank;
    int partnerrank;
    time_t start;
    TAILQ_ENTRY(qentry) pointers;
} qentry;


static TAILQ_HEAD(, qentry) head;

void enqueue(char** operation, char** datatype, int count, int datasize, char** communicator, int processrank, int partnerrank, time_t ctime){
    //printf("Operation: %s\n", *operation);
    //printf("Hier der Communicator: %s\n", *communicator);
    //printf("Hier der Type: %s\n", *datatype);
    //printf("Current Time: %ld \n", ctime);
    qentry *item = (qentry*)malloc(sizeof(qentry));
    item->operation = strdup(*operation);
    item->datatype = strdup(*datatype);
    item->count = count;
    item->datasize = datasize;
    item->communicator = strdup(*communicator);
    item->processrank = processrank;
    item-> partnerrank = partnerrank;
    item->start = ctime;
    TAILQ_INSERT_TAIL(&head, item, pointers);
}

qentry* dequeue(){
    qentry *item;
    item = TAILQ_FIRST(&head);
    TAILQ_REMOVE(&head, item, pointers);
    return item;
}

//Needs to be global!
pthread_t MONITOR_THREAD = NULL;

//Database Information
static MYSQL *conn;
static char *server = "192.168.42.9";
static char *user = "AnnaLena";
static char *password = "annalena";
static char *database = "DataFromMPI";

static const int LIMIT = 1000;
static int count = LIMIT;
static char *batchstring = "INSERT INTO MPI_Data(operation, datatype, count, datasize, communicator, processrank, partnerrank)VALUES";

static void insertData(char **batchstr){
    count = LIMIT;
    char *batch = *batchstr;
    batch[strlen(batch)-1]=';';
    //printf("%s\n", batch);
    if(mysql_query(conn, batch)){
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
}

static void collectData(qentry **item, char **batchstr){
    qentry *q = *item;
    int countlen = (q->count==0)?1:(int)log10(q->count)+1;
    int datasizelen = (q->datasize==0)?1:(int)log10(q->datasize)+1;
    int processranklen = (q->processrank==0)?1:(int)log10(q->processrank)+1;
    int partnerranklen = (q->partnerrank==0)?1:(int)log10(q->partnerrank)+1;
    //Speicherplatz für alle Einträge als Char + 3* '' für die Chars + 6* , und Leertaste + ()
    int datalen = strlen(q->operation) + strlen(q->datatype) + countlen + datasizelen + strlen(q->communicator) + processranklen + partnerranklen + 3*2 + 6*2 +2 +1;
    //printf("Datalen: %d\n", datalen);
    char *data=(char*)malloc(datalen);
    sprintf(data, "('%s', '%s', %d, %d, '%s', %d, %d),", q->operation, q->datatype, q->count, q->datasize, q->communicator, q->processrank, q->partnerrank);
    *batchstr = realloc(*batchstr, strlen(*batchstr)+1 + strlen(data)+1);
    strcat(*batchstr, data);
    free(data);
    count--;
    if(count==0){
	char *batch = *batchstr;
	insertData(&batch);
	*batchstr = realloc(*batchstr, strlen(batchstring)+1);
	strcpy(*batchstr, batchstring);
    }
}

static void* MonitorFunc(void* _arg){
    qentry *item = (qentry*)malloc(sizeof(qentry));
    char *batch=(char*) malloc(strlen(batchstring)+1);
    strcpy(batch, batchstring);
    int finish = 0;
    while(!finish){
        if(TAILQ_EMPTY(&head)){
            sleep(1);
            if(TAILQ_EMPTY(&head)){
                finish = 1; 
            }
        }
        else {
            item = dequeue();
            collectData(&item, &batch);
	    free(item);
            //printf("%d\n", item->data);
        }
    }
    free(batch);
}

static const char FUNC_NAME[] = "MPI_Init";


int MPI_Init(int *argc, char ***argv)
{
    //printf("test\n");   
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
