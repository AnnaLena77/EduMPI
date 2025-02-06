#include <time.h>

void intToBinary(int integer, char* buffer, int* offset);
void doubleToBinary(double value, char* buffer, int* offset);
void stringToBinary(char* string, char* buffer, int* offset);
void timestampToBinary(struct timespec time, char* buffer, int* offset, int round_seconds);
void byteaToBinary(int* array, int length, char* buffer, int* offset, int p2p);
void newRow(char* buffer, int column_count, int* offset);
//void createTrailer(char* buffer, int* offset);

extern char PGCOPY_HEADER[19];
extern char PGCOPY_TRAILER[2];
