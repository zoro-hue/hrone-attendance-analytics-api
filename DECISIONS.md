# DECISIONS.md

### 1. Indexes
Employees: unique `{ emp_code: 1 }` for lookups; `{ department: 1, emp_code: 1 }` for department filtering. Attendance: unique `{ emp_code: 1, date: 1 }` enforces single daily logs; `{ date: -1, emp_code: 1 }` covers paginated list queries and date ranges. Rejected: `{ date: 1 }` (redundant with compound prefix) and `status` alone (low selectivity).

### 2. Punch-in race
Both requests pass validation and call `insert_one`. WiredTiger serializes writes against the unique `{ emp_code: 1, date: 1 }` index. One succeeds (201); the second raises `DuplicateKeyError`, caught and returned as 409 Conflict.

### 3. Ties
Assigned via `$setWindowFields` with `$rank` on `total_late_minutes` descending. Tied employees receive identical ranks (e.g., both 10), and all qualifying records within limit are returned, broken secondarily by `emp_code` ascending.

### 4. Headcount
Headcount is computed directly from `employees` filtered by `joined_on <= end_of_month`. Attendance logs are joined onto employees. Employees with zero records are fully counted because base headcount stems from employee records, not attendance logs.

### 5. 100x Data Bottleneck
Raw log aggregations bottleneck CPU/memory. We would pre-aggregate daily departmental metrics into a rollups collection using Change Streams or scheduled `$merge`, turning analytical queries into O(1) point lookups.
