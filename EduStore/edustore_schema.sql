-- =========================================================
-- edustore_schema.sql
-- Schema for EduMPI / TimescaleDB (TigerData)

-- TimescaleDB Extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Tables

CREATE TABLE IF NOT EXISTS edumpi_runs (
    edumpi_run_id SERIAL PRIMARY KEY,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NULL,
    user_id VARCHAR NULL,
    program_name VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS edumpi_cluster_info (
    edumpi_cluster_id SERIAL PRIMARY KEY,
    edumpi_run_id INT REFERENCES edumpi_runs(edumpi_run_id),
    processorname TEXT NULL,
    processrank INT4 NULL
);

CREATE TABLE IF NOT EXISTS edumpi_running_data (
    edumpi_run_id INT REFERENCES edumpi_runs(edumpi_run_id),
    "function" TEXT NULL,
    communicationtype TEXT NULL,
    senddatasize BIGINT NULL,
    recvdatasize BIGINT NULL,
    communicationarea TEXT NULL,
    processorname TEXT NULL,
    processrank INT4 NULL,
    partnerrank INT4 NULL,
    coll_algorithm TEXT NULL,
    time_start TIMESTAMPTZ NOT NULL,
    time_end TIMESTAMPTZ NOT NULL,
    latesendertime DOUBLE PRECISION NULL,
    laterecvrtime DOUBLE PRECISION NULL,
    time_diff DOUBLE PRECISION NULL,
    coll_partnerranks BYTEA NULL
);

-- Hypertable

SELECT create_hypertable(
    'edumpi_running_data',
    'time_end',
    if_not_exists => TRUE
);

-- Continuous Aggregate:
-- edumpi_secondly_data

CREATE MATERIALIZED VIEW edumpi_secondly_data
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 second', time_end) AS time_end,
    time_start,
    edumpi_run_id,
    processrank,
    processorname,
    communicationtype,
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
    communicationtype;

SELECT add_continuous_aggregate_policy(
    'edumpi_secondly_data',
    start_offset => NULL,
    end_offset => NULL,
    schedule_interval => INTERVAL '1 second'
);

ALTER MATERIALIZED VIEW edumpi_secondly_data
SET (timescaledb.materialized_only = false);

CALL refresh_continuous_aggregate('edumpi_secondly_data', NULL, NULL);

-- Continuous Aggregate:
-- edumpi_detailed_data

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
    coll_algorithm,
    coll_partnerranks;

SELECT add_continuous_aggregate_policy(
    'edumpi_detailed_data',
    start_offset => NULL,
    end_offset => NULL,
    schedule_interval => INTERVAL '1 second'
);

ALTER MATERIALIZED VIEW edumpi_detailed_data
SET (timescaledb.materialized_only = false);

CALL refresh_continuous_aggregate('edumpi_detailed_data', NULL, NULL);
