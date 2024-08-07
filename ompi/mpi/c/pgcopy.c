#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "ompi/mpi/c/pgcopy/pgcopy.h"

/*
PGCOPY-Binary File:

HEADER (19 Bytes) consists of:
- 11-byte sequence: PGCOPY\n\377\r\n\0 
- Flags field (32 Bit mask, 4 Byte) for important aspects of the file format
- Bit16: 1 - OIDs are included / 0 - OIDs are not included (not supported anymore)
- Header extension area length (32 bit integer, 4 Byte) may contain a sequence of self-identifying chunks

In order to import the binary, it is sufficient to create a 19 byte array and fill it with the PGCOPY sequence

TUPLES (data area), every db-entry consists of:
- 16-bit (2 Byte) Integer-Count: Count of the number of fields in the tuple
- 32-bit (4 Byte) length word, specifies the size of the data to be written to the respective column (repeated for each field in the tuple)
- that many bytes of field data

no alignment padding or any other extra data between fields

FILE TRAILER
- 16-bit (2 Byte) integer word containing -1

DETAILS: https://www.postgresql.org/docs/current/sql-copy.html#AEN66736

*/

const char PGCOPY_HEADER[19] = {'P', 'G', 'C', 'O', 'P', 'Y', '\n', 0xff, '\r', '\n', '\0'};
const char PGCOPY_TRAILER[2] = {0xff, 0xff};

void intToBinary(int integer, char* buffer, int* offset){
    int off = *offset;
    
    buffer[off] = 0;
    buffer[off+1] = 0;
    buffer[off+2] = 0;
    buffer[off+3] = 4;
    
    off += 4;
    
    buffer[off] = (integer >> 24) & 0xff;
    buffer[off+1] = (integer >> 16) & 0xff;
    buffer[off+2] = (integer >> 8) & 0xff;
    buffer[off+3] = integer & 0xff;
    *offset += 8;
}

void doubleToBinary(double value, char* buffer, int* offset) {
    int off = *offset;

    buffer[off] = 0;
    buffer[off+1] = 0;
    buffer[off+2] = 0;
    buffer[off+3] = 8; // 8 bytes für double

    off += 4;

    // Zeiger auf den double-Wert als uint64_t behandeln
    uint64_t integer_value = *((uint64_t*)&value);
    
    buffer[off] = (integer_value >> 56) & 0xff;
    buffer[off+1] = (integer_value >> 48) & 0xff;
    buffer[off+2] = (integer_value >> 40) & 0xff;
    buffer[off+3] = (integer_value >> 32) & 0xff;
    buffer[off+4] = (integer_value >> 24) & 0xff;
    buffer[off+5] = (integer_value >> 16) & 0xff;
    buffer[off+6] = (integer_value >> 8) & 0xff;
    buffer[off+7] = integer_value & 0xff;
    
    *offset += 12;  // 4 bytes für die Länge + 8 bytes für den double-Wert
}

void stringToBinary(char* string, char* buffer, int* offset){
    int off = *offset;
    int len = strlen(string);
    
    buffer[off] = 0;
    buffer[off+1] = 0;
    buffer[off+2] = 0;
    buffer[off+3] = len;
    
    off += 4;
    
    memcpy(buffer + off, string, len);
    
    *offset += len+4;

}

void timestampToBinary(struct timespec time, char* buffer, int* offset, int round_seconds){
    int off = *offset;
    
    buffer[off] = 0;
    buffer[off+1] = 0;
    buffer[off+2] = 0;
    buffer[off+3] = 8;
    
    off += 4;
    *offset += 4;
    
    //int utc_offset_seconds = 2 * 3600;

    //count of seconds since 01.01.2000
    time_t seconds_since_2000 = time.tv_sec - 946684800; // + utc_offset_seconds;
    long long timestamp_micro;
    //cast into microseconds
    if(!round_seconds){
        long microseconds = (time.tv_nsec / 1000);
        timestamp_micro = seconds_since_2000 * 1000000LL + microseconds;
    } else {
        timestamp_micro = seconds_since_2000 * 1000000LL; // Milliseconds and nanoseconds set to 0
    }
    //timestamp_micro += utc_offset_seconds * 1000000LL;
    
    buffer[off] = (timestamp_micro >> 56) & 0xff;
    buffer[off+1] = (timestamp_micro >> 48) & 0xff;
    buffer[off+2] = (timestamp_micro >> 40) & 0xff;
    buffer[off+3] = (timestamp_micro >> 32) & 0xff;
    buffer[off+4] = (timestamp_micro >> 24) & 0xff;
    buffer[off+5] = (timestamp_micro >> 16) & 0xff;
    buffer[off+6] = (timestamp_micro >> 8) & 0xff;
    buffer[off+7] = timestamp_micro & 0xff;
    
    *offset += 8; 
}

void newRow(char* buffer, int column_count, int* offset){
    //memcpy(buffer, PGCOPY_HEADER, 19);
    //*offset += 19;
    
    buffer[*offset] = 0;
    buffer[*offset+1] = column_count;
    
    *offset += 2;
}

void createTrailer(char* buffer, int* offset){
    memcpy(buffer+*offset, PGCOPY_TRAILER, 2);
    *offset += 2;
}


