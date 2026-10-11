# Employee Attendance & Analytics API

A FastAPI + MongoDB backend service implementing employee attendance tracking, regularization audit trails, and analytics aggregation pipelines according to the business specifications (R1–R10).

## Tech Stack
- **Python 3.11+**
- **FastAPI**
- **MongoDB 6.0+** with **PyMongo**
- **Pydantic v2**

---

## Quickstart

### 1. Prerequisites & Environment
Ensure MongoDB is running locally on port 27017 (or provide a remote connection string).

```bash
# Clone and enter directory
git clone https://github.com/zoro-hue/hrone-attendance-analytics-api.git
cd hrone-attendance-analytics-api

# Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
The application reads `MONGO_URI` and `MONGO_DB` from environment variables (or `.env` file):
```bash
MONGO_URI=mongodb://localhost:27017
MONGO_DB=attendance_db
```

### 3. Seed Sample Data (Optional)
```bash
python sample_seed.py
```

### 4. Run the API Server
```bash
uvicorn app.main:app --port 8000
```
- Interactive Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 5. Run the Automated Test Suite
Running the automated integration test suite requires an accessible, compatible MongoDB instance (MongoDB 6.0+ is recommended for `$densify` and `$setWindowFields` support).

> **Warning**: The test suite uses the database specified by `MONGO_DB` (defaulting to `attendance_test_db` during test runs). Test fixtures clean up test data by calling `delete_many({})` between runs. Do not point the test environment to a production or non-test database.

```bash
pip install pytest httpx
python -m pytest tests/test_api.py -v
```

---

## Key Features & Architectural Highlights

1. **Strict Business Rule Implementation (R1–R10)**:
   - **R1 (Timezone & Overnight Shifts)**: Calculations are anchored to IST (UTC+05:30). Overnight shifts map pre-dawn punches to the starting calendar day. Timestamps are truncated to whole seconds.
   - **R2 (Late Arrivals)**: 10-minute grace period with full minute floor calculation when late.
   - **R3 (Overtime)**: Floor minutes past shift end with a 30-minute qualification threshold.
   - **R4 (Work Hours)**: Decimal-accurate rounding using half-up arithmetic (`ROUND_HALF_UP`).
   - **R5 (Half-Day Threshold)**: Automatically flagged when work hours < 4.50 hours.
   - **R6 & R7 (Presence & Working Days)**: Differentiates working days from non-working days; counts mid-month joiners correctly.
   - **R8 (Decimal Representation)**: Strict 2 decimal places for hours/counts and 4 decimal places for rates.
   - **R9 (Headcount)**: Point-in-time headcount by join date regardless of presence in attendance logs.
   - **R10 (Pagination)**: 1-indexed pagination with accurate total counts.

2. **Concurrency Safe**:
   - Unique compound index `(emp_code, date)` guarantees race-safe punch-ins with `409 Conflict` on duplicates.
   - Atomic document updates ensure concurrent regularizations do not lose history audit logs.

3. **MongoDB Aggregation Pipelines**:
   - Database-side aggregation pipelines (`$lookup`, `$group`, `$setWindowFields`, `$densify`).
   - Pure database calculations without in-memory dataset iteration.
   - Efficient compound index usage and offset/limit pagination (`skip`/`limit`).

4. **Query Plan Explainability**:
   - `GET /admin/explain/{endpoint}` executes MongoDB's `explain('executionStats')` on the exact queries and aggregation pipelines used by the operational endpoints.

---

## Project Structure
```
hrone-attendance-analytics-api/
├── app/
│   ├── __init__.py
│   └── main.py          # Complete application implementation in a single cohesive file
├── sample_data/         # MongoDB Extended JSON sample records
├── tests/
│   ├── __init__.py
│   └── test_api.py      # Automated pytest suite (99 unit, integration, and concurrency regression tests)
├── DATA_MODEL.md        # Document schemas and sample record reference
├── DECISIONS.md         # Architecture, indexing, and concurrency rationale (197 words)
├── openapi.yaml         # Complete OpenAPI 3.0 contract
├── README.md            # Setup and operations documentation
├── requirements.txt     # Python dependencies
├── REVIEW.md            # Code review documenting 25 starter defects and remedies
└── sample_seed.py       # Sample data loader script
```
