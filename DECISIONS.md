# DECISIONS.md

### 1. Indexes
- **`employees`**:
  - `{ emp_code: 1 }` (unique): Guarantees employee identity uniqueness and fast point lookups.
  - `{ department: 1, emp_code: 1 }`: Supports department filtering, paginated employee listing, and departmental headcount counts.
- **`attendance_logs`**:
  - `{ emp_code: 1, date: 1 }` (unique): Enforces the core business invariant of at most one record per employee per day, and backs `employee_monthly` range queries.
  - `{ date: -1, emp_code: 1 }`: Covers paginated `attendance_list` queries without in-memory sorting.
  - `{ date: 1 }`: Powers month-range `$match` stages across analytics pipelines.
- **Considered and rejected**: An index on `status` alone (low cardinality, minimal selectivity) and TTL indexes (attendance records are permanent audit data).

### 2. Punch-in race
When two identical requests arrive simultaneously, both pass input validation and attempt `insert_one`. MongoDB's WiredTiger engine serializes the write against the `{ emp_code: 1, date: 1 }` unique index. Exactly one thread succeeds and returns `201 Created`. The second thread violates the index constraint, raising a `DuplicateKeyError` (code 11000), which our handler catches and converts into HTTP `409 Conflict`. No duplicate record can ever be created.

### 3. Ties
The leaderboard computes competition ranks using `$setWindowFields` with `$rank` sorted by `total_late_minutes` descending. The filter `rank <= limit` is applied *after* ranking. If two employees tie at rank 10 with `limit=10`, both share rank 10 and both are returned (yielding 11 items), sorted secondarily by `emp_code` ascending. Array indexing is never used for rank.

### 4. Headcount
The pipeline bases department headcount on the `employees` collection filtered by `joined_on <= end_of_month`. It joins attendance records onto this base rather than deriving headcount from attendance logs. Consequently, employees who joined on or before the period end are counted even if they have zero attendance records.

### 5. One thing to change for 100x data
At 100x volume (~10 million logs), we would compute daily/monthly attendance summaries asynchronously via MongoDB Change Streams or an incremental daily `$merge` pipeline into a `daily_rollups` collection. Analytical queries would read pre-aggregated buckets instead of running multi-stage aggregations over raw punch records.
