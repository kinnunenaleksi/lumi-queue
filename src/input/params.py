# Original column names map
column_dict = {
    "Account": "account_id",
    "UID": "user_id",
    "ReservationId": "reservation_id",
    "JobIDRaw": "job_id",
    "Priority": "priority",
    "Partition": "partition",
    "State": "state",
    "Submit": "submit_ts",
    "Eligible": "eligible_start_ts",
    "Start": "start_ts",
    "End": "end_ts",
    "ElapsedRaw": "elapsed_seconds",
    "TimelimitRaw": "timelimit_minutes",
    "AllocTRES": "allocated_resources",
}

# Partitions in Scope
USED_PARTITIONS = ["small", "small-g", "standard", "standard-g", "largemem", "lumid"]

# Columns that cannot be nulls
NON_NA_COLUMNS = [
    "submit_ts",
    "eligible_start_ts",
    "start_ts",
    "end_ts",
    "state",
    "priority",
    "timelimit_minutes",
]

# These states are dropped, as the run-time is always 0
INVALID_STATES = []#["REQUEUED", "PENDING"]

