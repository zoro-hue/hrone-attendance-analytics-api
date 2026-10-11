# DECISIONS.md

### 1. Indexes
Employees: unique `{ emp_code: 1 }`; `{ department: 1, emp_code: 1 }` for filtering. Attendance: unique `{ emp_code: 1, date: 1 }` prevents duplicates; `{ date: -1, emp_code: 1 }` covers paginated listings and date ranges. Rejected: `{ date: 1 }` (compound prefix covers it) and `{ status: 1 }` (low selectivity).

### 2. Punch-in race
WiredTiger serializes concurrent writes against unique `{ emp_code: 1, date: 1 }`. First insert succeeds (201); concurrent insert raises `DuplicateKeyError`, returning 409 Conflict without check-then-insert races.

### 3. Ties
Using `$setWindowFields` with `$rank` on `total_late_minutes` descending. Ties share rank (skipping subsequent ranks). Filtering by `rank <= limit` returns all tied records, meaning responses exceed `limit` rows when ties straddle the cutoff, ordered secondarily by `emp_code` ascending.

### 4. Headcount
Headcount is aggregated from `employees` matching `joined_on <= end_of_month`. Attendance metrics are aggregated from `attendance_logs` and joined in application logic, ensuring employees with zero attendance records are fully counted in departmental headcount.

### 5. 100x Data Bottleneck
Raw log aggregations bottleneck CPU and memory. We would pre-aggregate daily departmental metrics into a rollups collection via Change Streams or scheduled `$merge`, converting complex analytics into O(1) point lookups.
