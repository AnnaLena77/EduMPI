-- Add the communicator (communicationarea) to the two continuous aggregates, so that the GUI can
-- show which processes take part in which communicator.
--
-- Run as the owner of the tables (postgres), outside of a transaction block:
--     psql -U postgres -d <database> -f 001_communicator_in_aggregates.sql
--
-- What happens:
--   * The aggregates are dropped and created again. TimescaleDB cannot add a column to the GROUP BY of
--     an existing continuous aggregate. The raw table edumpi_running_data is NOT touched.
--   * Until the refresh at the end has caught up, the GUI shows nothing for old runs. Run it when no
--     course session is running. The refresh reads the whole raw table (about 47 million rows in
--     November 2026).
--   * Older GUI versions keep working: their queries sum over the new column.
--   * Rows written before the EduMPI change that stores a label for every communicator have an empty
--     label (or MPI_COMM_WORLD); the GUI treats an empty label as "unknown".
-- The settings below are those of the live database on 2026-11-04 (materialized_only = true, refresh
-- every second, no offsets); the repo file edustore_schema.sql said materialized_only = false.

DROP MATERIALIZED VIEW IF EXISTS edumpi_detailed_data;
DROP MATERIALIZED VIEW IF EXISTS edumpi_secondly_data;

CREATE MATERIALIZED VIEW edumpi_secondly_data
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', time_end) AS time_end,
    time_start,
    edumpi_run_id,
    processrank,
    processorname,
    communicationtype,
    communicationarea,
    SUM(senddatasize) AS send_ds,
    SUM(recvdatasize) AS recv_ds,
    SUM(latesendertime) AS latesendertime,
    SUM(laterecvrtime) AS laterecvrtime,
    SUM(time_diff) AS time_diff
FROM edumpi_running_data
GROUP BY
    time_bucket('1 second', time_end),
    time_start,
    edumpi_run_id,
    processrank,
    processorname,
    communicationtype,
    communicationarea
WITH NO DATA;

CREATE MATERIALIZED VIEW edumpi_detailed_data
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', time_end) AS time_end,
    time_start,
    edumpi_run_id,
    "function",
    processrank,
    partnerrank,
    SUM(senddatasize) AS send_ds,
    SUM(recvdatasize) AS recv_ds,
    SUM(time_diff) AS comm_time,
    communicationtype,
    communicationarea,
    coll_algorithm,
    coll_partnerranks
FROM edumpi_running_data
GROUP BY
    time_bucket('1 second', time_end),
    time_start,
    edumpi_run_id,
    "function",
    processrank,
    partnerrank,
    communicationtype,
    communicationarea,
    coll_algorithm,
    coll_partnerranks
WITH NO DATA;

ALTER MATERIALIZED VIEW edumpi_secondly_data SET (timescaledb.materialized_only = true);
ALTER MATERIALIZED VIEW edumpi_detailed_data SET (timescaledb.materialized_only = true);

GRANT SELECT, INSERT, UPDATE ON edumpi_secondly_data, edumpi_detailed_data TO edumpi;

SELECT add_continuous_aggregate_policy('edumpi_secondly_data',
    start_offset => NULL, end_offset => NULL, schedule_interval => INTERVAL '1 second');
SELECT add_continuous_aggregate_policy('edumpi_detailed_data',
    start_offset => NULL, end_offset => NULL, schedule_interval => INTERVAL '1 second');

-- must run outside of a transaction block
CALL refresh_continuous_aggregate('edumpi_secondly_data', NULL, NULL);
CALL refresh_continuous_aggregate('edumpi_detailed_data', NULL, NULL);
