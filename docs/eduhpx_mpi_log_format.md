# EduHPX MPI log format, version 1

The measuring MPI (branch `eduhpx_mpi_data` of EduMPI) writes one binary file per rank:
`$EDUHPX_MPI_LOG_DIR/eduhpx.mpi.<rank>.bin` (the current directory if the variable is not set). The file is written when
event segments are full and at `MPI_Finalize` (also at normal process exit if the program never reaches `MPI_Finalize`). All numbers
are **little endian**. All times are **wall clock (`CLOCK_REALTIME`) in nanoseconds since 1970-01-01 UTC, not rounded**; `0` means "not set".
The clocks of different nodes are not synchronised; the merge corrects them with measured offsets.

## Layout

```
header   64 bytes
records  n * 64 bytes
trailer  16 + nf * 32 + nc * 64 bytes   (n = (trailer_offset - 64) / 64)
```

### Header (64 bytes)

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `EDUHPXM1` |
| 8 | 4 | format version = 1 |
| 12 | 4 | record size = 64 |
| 16 | 4 | `rank` (rank in `MPI_COMM_WORLD`, i32) |
| 20 | 4 | `world_size` (i32) |
| 24 | 32 | node name (`MPI_Get_processor_name`, zero terminated) |
| 56 | 8 | `trailer_offset` (u64), `0` while the file is open (a file without a trailer, e.g. after a crash, holds records only: read until the end of the file, function and communicator ids cannot be resolved) |

### Record (64 bytes)

One record per MPI call (all wrapped calls are recorded) in the order the segments were written; sort by `start_ns` or use `seq`.

| offset | size | field | meaning |
|---|---|---|---|
| 0 | 8 | `start_ns` u64 | start of the call. For `MPI_Irecv`: the time the receive was **posted**. |
| 8 | 8 | `end_ns` u64 | the call returned (`MPI_Irecv`/`MPI_Isend` return before the request is complete) |
| 16 | 8 | `done_ns` u64 | only if flag `HAS_DONE` (never for collectives; a send that completed before `MPI_Isend` returned has none): `MPI_Irecv`: the time the message **arrived** (the request completed); `MPI_Isend` of a send that was still in flight when the call returned: the time the send **request completed**. Blocking `MPI_Recv` in ob1: arrival. |
| 24 | 8 | `send_bytes` i64 | bytes sent |
| 32 | 8 | `recv_bytes` i64 | bytes received (`MPI_Irecv`: filled in when the request completes) |
| 40 | 4 | `partner` i32 | partner rank in `MPI_COMM_WORLD` (collectives: the last partner the call talked to); `-1` unknown or a receive that never completed; for `MPI_ANY_SOURCE` the real sender once the message arrived (flag `ANY_SOURCE`) |
| 44 | 2 | `comm` u16 | index into the communicator table of the trailer, `0xFFFF` none |
| 46 | 2 | `func` u16 | index into the function table of the trailer (name of the MPI call), `0xFFFF` if the table was full |
| 48 | 4 | `flags` u32 | bit 0 `BLOCKING`, 1 `COLLECTIVE` (not point-to-point), 2 `NO_END` (the call had not returned at the end), 3 `HAS_DONE`, 4 `ANY_SOURCE`, 5 `NOT_COMPLETED` (the request had not completed at the end, e.g. an idle posted receive) |
| 52 | 4 | `sendcount` u32 | element count sent |
| 56 | 4 | `recvcount` u32 | element count received |
| 60 | 4 | `seq` u32 | event number of this rank (order of the calls, counted when the call started; wraps after 2^32) |

### Trailer

```
8   magic "EDUHPXT1"
4   nf   number of function names
4   nc   number of communicator labels
nf * 32   function names, zero padded
nc * 64   communicator labels, zero padded ("MPI_COMM_WORLD", "MPI COMM 3 DUP FROM 0", "... @<world rank of the first member>")
```

## What differs from the database version of EduMPI

- No rounding of the start time to seconds; full nanoseconds for start, end and done.
- No separate `MPI_Isend_done` row: the completion time of a send is `done_ns` of the `MPI_Isend` record. The post time of a receive
  is `start_ns` and its arrival is `done_ns` of the `MPI_Irecv` record (the database version overwrote the start with the arrival).
- Not exported (kept only in the database version): late-sender/late-receiver times, algorithm, send mode, BTL/protocol, the
  partner bit mask of collectives, data type and operation names.
- Hooks always on, for the UCX and the ob1 PML: receive post and arrival time, real source of `MPI_ANY_SOURCE`, completion time of a
  send that is still in flight when `MPI_Isend` returns (`EDUHPX_MPI_SEND_COMPLETION=0` turns the send hook off).
- Memory: a segment of 2048 events stays in memory while it holds an entry that is not complete (call not returned, request not
  completed). Everything else is written when the next segment is started.

## Reading

`tools/eduhpx_dump.py <file>` (Python 3, standard library only) prints the header and the records as CSV.
