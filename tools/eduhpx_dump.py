#!/usr/bin/env python3
"""Dump an EduHPX MPI log file (eduhpx.mpi.<rank>.bin): header, then one CSV row per record.
usage: eduhpx_dump.py [--summary] file [file ...]   (format: docs/eduhpx_mpi_log_format.md)"""
import collections
import struct
import sys

FLAGS = ["BLOCKING", "COLLECTIVE", "NO_END", "HAS_DONE", "ANY_SOURCE", "NOT_COMPLETED"]
REC = struct.Struct("<QQQqqiHHIIII")  # 64 bytes


def read(path):
    data = open(path, "rb").read()
    if data[:8] != b"EDUHPXM1":
        raise SystemExit("%s: not an EduHPX MPI log (magic %r)" % (path, data[:8]))
    version, rec_size, rank, world = struct.unpack_from("<IIii", data, 8)
    node = data[24:56].split(b"\0")[0].decode()
    (trailer,) = struct.unpack_from("<Q", data, 56)
    if version != 1 or rec_size != 64 or REC.size != 64:
        raise SystemExit("%s: unsupported version %d / record size %d" % (path, version, rec_size))
    end = trailer if trailer else len(data) - (len(data) - 64) % 64
    funcs, comms = [], []
    if trailer:
        if data[trailer:trailer + 8] != b"EDUHPXT1":
            raise SystemExit("%s: bad trailer" % path)
        nf, nc = struct.unpack_from("<II", data, trailer + 8)
        p = trailer + 16
        funcs = [data[p + 32 * i:p + 32 * i + 32].split(b"\0")[0].decode() for i in range(nf)]
        p += 32 * nf
        comms = [data[p + 64 * i:p + 64 * i + 64].split(b"\0")[0].decode() for i in range(nc)]
    recs = []
    for off in range(64, end, 64):
        (start, endt, done, sb, rb, partner, comm, func, flags, sc, rc, seq) = REC.unpack_from(data, off)
        recs.append(dict(start_ns=start, end_ns=endt, done_ns=done, send_bytes=sb, recv_bytes=rb, partner=partner,
                         comm=comms[comm] if comm < len(comms) else "", func=funcs[func] if func < len(funcs) else "?%d" % func,
                         flags=flags, sendcount=sc, recvcount=rc, seq=seq))
    return dict(rank=rank, world=world, node=node, trailer=trailer, records=recs, size=len(data))


def flag_names(f):
    return "|".join(n for i, n in enumerate(FLAGS) if f >> i & 1)


def main():
    args = sys.argv[1:]
    summary = "--summary" in args
    files = [a for a in args if a != "--summary"]
    if not files:
        raise SystemExit(__doc__)
    for path in files:
        d = read(path)
        print("# %s rank=%d world=%d node=%s records=%d bytes=%d trailer=%s" % (path, d["rank"], d["world"], d["node"], len(d["records"]), d["size"], "yes" if d["trailer"] else "NO"))
        if summary:
            c = collections.Counter(r["func"] for r in d["records"])
            print("#   " + ", ".join("%s=%d" % kv for kv in c.most_common()))
            f = collections.Counter(n for r in d["records"] for i, n in enumerate(FLAGS) if r["flags"] >> i & 1)
            print("#   flags: " + ", ".join("%s=%d" % kv for kv in sorted(f.items())))
            continue
        print("func,comm,partner,start_ns,end_ns,done_ns,send_bytes,recv_bytes,sendcount,recvcount,flags,seq")
        for r in sorted(d["records"], key=lambda r: r["start_ns"]):
            print("%s,\"%s\",%d,%d,%d,%d,%d,%d,%d,%d,%s,%d" % (r["func"], r["comm"], r["partner"], r["start_ns"], r["end_ns"], r["done_ns"],
                  r["send_bytes"], r["recv_bytes"], r["sendcount"], r["recvcount"], flag_names(r["flags"]), r["seq"]))


if __name__ == "__main__":
    main()
