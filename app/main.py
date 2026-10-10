"""
Employee Attendance & Analytics API
Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import json
import math
import os
import re
import calendar
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Any

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, Field, field_validator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

load_dotenv()


def _clean_bson(doc: Any) -> Any:
    """Serialize and deserialize BSON types (Binary, UUID, ObjectId) so FastAPI handles them cleanly."""
    return json.loads(json_util.dumps(doc))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

# Epoch-ms boundaries (contract: below 100_000_000_000 is rejected)
EPOCH_MS_MIN = 100_000_000_000
EPOCH_MS_MAX = 4_102_444_800_000

PRESENCE_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = {"PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"}

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
client: MongoClient = None  # type: ignore
db = None


def get_db():
    return db


def _create_indexes():
    """Create all required indexes idempotently at startup."""
    _db = get_db()
    # Unique index on emp_code — enforces concurrency-safe uniqueness
    _db.employees.create_index([("emp_code", ASCENDING)], unique=True)
    # For department filter + listing
    _db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)])

    # Unique compound index — enforces one record per (emp_code, date)
    _db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True
    )
    # For list attendance sorted by date desc, emp_code asc + date range filters
    _db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)]
    )
    # For analytics that filter by date range
    _db.attendance_logs.create_index([("date", ASCENDING)])


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI):
    global client, db
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGO_DB", "attendance_db")]
    _create_indexes()
    yield
    client.close()


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Rounding helpers (R8: half-up, not banker's rounding)
# ---------------------------------------------------------------------------

def round_half_up(value: float, decimals: int = 2) -> float:
    """Round to `decimals` places using ROUND_HALF_UP."""
    if value is None:
        return None  # type: ignore
    d = Decimal(str(value))
    quant = Decimal(10) ** -decimals
    return float(d.quantize(quant, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def epoch_ms_to_datetime(ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC-aware datetime, truncated to whole seconds (R1)."""
    # Truncate to whole seconds
    ms_truncated = (ms // 1000) * 1000
    return datetime.fromtimestamp(ms_truncated / 1000, tz=timezone.utc)


def datetime_to_epoch_ms(dt: datetime) -> Optional[int]:
    """Convert a datetime to epoch milliseconds. Returns None for None."""
    if dt is None:
        return None
    return int(dt.timestamp() * 1000)


def _ensure_aware(dt: datetime) -> datetime:
    """If a datetime is naive, assume UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_shift_time(shift_str: str) -> tuple[int, int]:
    """Parse 'HH:MM' to (hour, minute)."""
    h, m = map(int, shift_str.split(":"))
    return h, m


def _is_overnight_shift(shift_start: str, shift_end: str) -> bool:
    """Check if shift_end <= shift_start (overnight shift per R1)."""
    sh, sm = _parse_shift_time(shift_start)
    eh, em = _parse_shift_time(shift_end)
    return (eh, em) <= (sh, sm)


def compute_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    """
    Determine attendance date per R1.
    - Convert punch_in to IST.
    - For overnight shifts: if punch_in IST time is earlier than shift_end,
      the record belongs to the previous day.
    - Otherwise, the attendance date is the IST calendar date of punch_in.
    """
    punch_in_ist = punch_in_utc.astimezone(IST)
    att_date = punch_in_ist.date()

    if _is_overnight_shift(shift_start, shift_end):
        eh, em = _parse_shift_time(shift_end)
        punch_time = (punch_in_ist.hour, punch_in_ist.minute, punch_in_ist.second)
        shift_end_time = (eh, em, 0)
        if punch_time < shift_end_time:
            att_date = att_date - timedelta(days=1)

    return att_date.isoformat()


def compute_late_minutes(punch_in_utc: datetime, shift_start: str, att_date_str: str) -> int:
    """
    Compute late minutes per R2.
    Grace period: 10 minutes. Late only if strictly more than 10:00 after shift_start.
    When late, late_minutes = floor(seconds since shift_start / 60).
    """
    sh, sm = _parse_shift_time(shift_start)
    att_date = date.fromisoformat(att_date_str)

    # Shift start is on the attendance date, in IST
    shift_start_dt = datetime(
        att_date.year, att_date.month, att_date.day,
        sh, sm, 0, tzinfo=IST
    )

    punch_in_ist = punch_in_utc.astimezone(IST)
    diff_seconds = (punch_in_ist - shift_start_dt).total_seconds()

    if diff_seconds <= 600:  # 10 * 60 = 600 seconds grace
        return 0
    # Floor of minutes since shift_start
    return int(diff_seconds // 60)


def compute_shift_end_dt(att_date_str: str, shift_start: str, shift_end: str) -> datetime:
    """Get the shift end as a datetime. For overnight shifts, shift_end is on the next day."""
    att_date = date.fromisoformat(att_date_str)
    eh, em = _parse_shift_time(shift_end)

    end_date = att_date
    if _is_overnight_shift(shift_start, shift_end):
        end_date = att_date + timedelta(days=1)

    return datetime(end_date.year, end_date.month, end_date.day, eh, em, 0, tzinfo=IST)


def compute_overtime(punch_out_utc: datetime, att_date_str: str, shift_start: str, shift_end: str) -> int:
    """
    Compute overtime per R3.
    overtime_minutes = floor(minutes between shift_end and punch_out), only if >= 30.
    """
    shift_end_dt = compute_shift_end_dt(att_date_str, shift_start, shift_end)
    punch_out_ist = punch_out_utc.astimezone(IST)

    diff_seconds = (punch_out_ist - shift_end_dt).total_seconds()
    if diff_seconds < 0:
        return 0
    minutes = int(diff_seconds // 60)
    return minutes if minutes >= 30 else 0


def compute_work_hours(punch_in_utc: datetime, punch_out_utc: datetime) -> float:
    """Compute work_hours per R4: seconds / 3600, 2 decimals half-up."""
    diff_seconds = (punch_out_utc - punch_in_utc).total_seconds()
    return round_half_up(diff_seconds / 3600, 2)


def compute_half_day(work_hours: float) -> bool:
    """Compute half_day per R5: true if rounded work_hours < 4.50."""
    return work_hours < 4.50


# ---------------------------------------------------------------------------
# Serialization: MongoDB doc → API response
# ---------------------------------------------------------------------------

def serialize_employee(doc: dict) -> dict:
    """Convert an employee MongoDB doc to API response format."""
    result = {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": datetime_to_epoch_ms(_ensure_aware(doc["created_at"])),
    }
    return result


def serialize_attendance(doc: dict) -> dict:
    """Convert an attendance_logs MongoDB doc to API response format."""
    punch_in = doc.get("punch_in")
    punch_out = doc.get("punch_out")

    history = doc.get("history") or []
    serialized_history = []
    for entry in history:
        h_entry = {
            "at": datetime_to_epoch_ms(_ensure_aware(entry["at"])),
            "by": entry["by"],
            "reason": entry["reason"],
            "changes": {},
        }
        for field_name, change in entry.get("changes", {}).items():
            from_val = change.get("from")
            to_val = change.get("to")
            # punch_in and punch_out changes store datetimes in MongoDB
            if field_name in ("punch_in", "punch_out"):
                from_val = datetime_to_epoch_ms(_ensure_aware(from_val)) if isinstance(from_val, datetime) else from_val
                to_val = datetime_to_epoch_ms(_ensure_aware(to_val)) if isinstance(to_val, datetime) else to_val
            h_entry["changes"][field_name] = {"from": from_val, "to": to_val}
        serialized_history.append(h_entry)

    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": datetime_to_epoch_ms(_ensure_aware(punch_in)) if isinstance(punch_in, datetime) else punch_in,
        "punch_out": datetime_to_epoch_ms(_ensure_aware(punch_out)) if isinstance(punch_out, datetime) else punch_out,
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": serialized_history,
    }


# ---------------------------------------------------------------------------
# Pydantic request models with validation
# ---------------------------------------------------------------------------

def _validate_epoch_ms(v: Any, field_name: str) -> int:
    """Validate that v is a valid epoch-ms integer."""
    if isinstance(v, bool):
        raise HTTPException(422, detail=f"{field_name} must be an integer epoch-ms value")
    if isinstance(v, float):
        raise HTTPException(422, detail=f"{field_name} must be an integer, not a float")
    if not isinstance(v, int):
        raise HTTPException(422, detail=f"{field_name} must be an integer epoch-ms value")
    if v < EPOCH_MS_MIN or v > EPOCH_MS_MAX:
        raise HTTPException(422, detail=f"{field_name} out of range [{EPOCH_MS_MIN}, {EPOCH_MS_MAX}]")
    return v


class EmployeeCreate(BaseModel):
    emp_code: str
    name: str
    email: str
    department: str
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("emp_code")
    @classmethod
    def validate_emp_code(cls, v: str) -> str:
        if not re.match(r"^EMP\d{4,6}$", v):
            raise ValueError("emp_code must match pattern EMP followed by 4-6 digits")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v) > 100:
            raise ValueError("name must be 1-100 characters")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if len(v) > 120 or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Invalid email format")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        if not v or len(v) > 50:
            raise ValueError("department must be 1-50 characters")
        return v

    @field_validator("shift_start", "shift_end")
    @classmethod
    def validate_shift_time(cls, v: str) -> str:
        if not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", v):
            raise ValueError("Shift time must be HH:MM (24-hour)")
        return v

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except (ValueError, TypeError):
            raise ValueError("joined_on must be YYYY-MM-DD")
        return v


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in PRESENCE_STATUSES:
            raise ValueError(f"status must be one of {sorted(PRESENCE_STATUSES)}")
        return v

    @field_validator("punched_at")
    @classmethod
    def validate_punched_at(cls, v):
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("punched_at must be an integer epoch-ms value")
            if isinstance(v, float):
                raise ValueError("punched_at must be an integer, not a float")
            if not isinstance(v, int):
                raise ValueError("punched_at must be an integer epoch-ms value")
            if v < EPOCH_MS_MIN or v > EPOCH_MS_MAX:
                raise ValueError(f"punched_at out of range [{EPOCH_MS_MIN}, {EPOCH_MS_MAX}]")
        return v


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None

    @field_validator("punched_at")
    @classmethod
    def validate_punched_at(cls, v):
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("punched_at must be an integer epoch-ms value")
            if isinstance(v, float):
                raise ValueError("punched_at must be an integer, not a float")
            if not isinstance(v, int):
                raise ValueError("punched_at must be an integer epoch-ms value")
            if v < EPOCH_MS_MIN or v > EPOCH_MS_MAX:
                raise ValueError(f"punched_at out of range [{EPOCH_MS_MIN}, {EPOCH_MS_MAX}]")
        return v


class RegularizeRequest(BaseModel):
    reason: str
    regularized_by: str
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        if len(v) < 5 or len(v) > 200:
            raise ValueError("reason must be 5-200 characters")
        return v

    @field_validator("regularized_by")
    @classmethod
    def validate_regularized_by(cls, v: str) -> str:
        if len(v) < 1 or len(v) > 50:
            raise ValueError("regularized_by must be 1-50 characters")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        if v is not None and v not in ALL_STATUSES:
            raise ValueError(f"status must be one of {sorted(ALL_STATUSES)}")
        return v

    @field_validator("punch_in", "punch_out")
    @classmethod
    def validate_punch_times(cls, v):
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("must be an integer epoch-ms value")
            if isinstance(v, float):
                raise ValueError("must be an integer, not a float")
            if not isinstance(v, int):
                raise ValueError("must be an integer epoch-ms value")
            if v < EPOCH_MS_MIN or v > EPOCH_MS_MAX:
                raise ValueError(f"out of range [{EPOCH_MS_MIN}, {EPOCH_MS_MAX}]")
        return v


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_month(month: str) -> tuple[str, str]:
    """Validate YYYY-MM format and return (first_day, last_day) as YYYY-MM-DD strings."""
    if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", month):
        raise HTTPException(422, detail="month must be YYYY-MM format")
    year, mon = map(int, month.split("-"))
    _, last = calendar.monthrange(year, mon)
    return f"{year:04d}-{mon:02d}-01", f"{year:04d}-{mon:02d}-{last:02d}"


def _validate_date(d: str) -> date:
    """Parse and validate a YYYY-MM-DD date string."""
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        raise HTTPException(422, detail=f"Invalid date format: {d}")


def _validate_page_params(page: int, page_size: int):
    """Validate pagination parameters per R10."""
    if page < 1:
        raise HTTPException(422, detail="page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(422, detail="page_size must be 1-100")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Readiness probe — returns 200 only if MongoDB answers a ping."""
    try:
        get_db().command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(503, detail="MongoDB is not available")


# ---------------------------------------------------------------------------
# POST /employees
# ---------------------------------------------------------------------------
@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    _db = get_db()
    if body.shift_start == body.shift_end:
        raise HTTPException(422, detail="shift_start must differ from shift_end")

    doc = body.model_dump()
    doc["created_at"] = datetime.now(tz=timezone.utc)

    try:
        _db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, detail="emp_code already exists")

    doc.pop("_id", None)
    return serialize_employee(doc)


# ---------------------------------------------------------------------------
# GET /employees
# ---------------------------------------------------------------------------
@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    _db = get_db()
    q: dict = {}
    if department:
        q["department"] = department

    total = _db.employees.count_documents(q)
    skip = (page - 1) * page_size
    items = list(
        _db.employees.find(q, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    return {
        "items": [serialize_employee(doc) for doc in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# POST /attendance/punch-in
# ---------------------------------------------------------------------------
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    _db = get_db()
    emp = _db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    if body.punched_at is not None:
        ts = epoch_ms_to_datetime(body.punched_at)
    else:
        # Current time, truncated to whole seconds
        now = datetime.now(tz=timezone.utc)
        ts = now.replace(microsecond=0)

    att_date = compute_attendance_date(ts, emp["shift_start"], emp["shift_end"])
    late_min = compute_late_minutes(ts, emp["shift_start"], att_date)

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_min,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        _db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, detail="Already punched in for this date")

    doc.pop("_id", None)
    return serialize_attendance(doc)


# ---------------------------------------------------------------------------
# POST /attendance/punch-out
# ---------------------------------------------------------------------------
@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    _db = get_db()
    emp = _db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    if body.punched_at is not None:
        ts = epoch_ms_to_datetime(body.punched_at)
    else:
        now = datetime.now(tz=timezone.utc)
        ts = now.replace(microsecond=0)

    # Find the most recent record whose punch_in is at or before punched_at
    record = _db.attendance_logs.find_one(
        {
            "emp_code": body.emp_code,
            "punch_in": {"$ne": None, "$lte": ts},
        },
        sort=[("punch_in", DESCENDING)],
    )

    if not record:
        raise HTTPException(404, detail="No punch-in found for this employee")

    if record.get("punch_out") is not None:
        raise HTTPException(409, detail="That record is already punched out")

    punch_in_dt = _ensure_aware(record["punch_in"])

    # Validate: punched_at must be > punch_in
    if ts <= punch_in_dt:
        raise HTTPException(422, detail="punch_out must be after punch_in")

    # Validate: punch_out within 24 hours of punch_in
    if (ts - punch_in_dt).total_seconds() > 86400:
        raise HTTPException(422, detail="punch_out must be within 24 hours of punch_in")

    wh = compute_work_hours(punch_in_dt, ts)
    ot = compute_overtime(ts, record["date"], emp["shift_start"], emp["shift_end"])
    hd = compute_half_day(wh)

    # Atomic update — only updates if punch_out is still None (concurrency guard)
    result = _db.attendance_logs.update_one(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts,
                "work_hours": wh,
                "overtime_minutes": ot,
                "half_day": hd,
            }
        },
    )

    if result.modified_count == 0:
        raise HTTPException(409, detail="Already punched out (concurrent request)")

    updated = _db.attendance_logs.find_one({"_id": record["_id"]})
    return serialize_attendance(updated)


# ---------------------------------------------------------------------------
# GET /attendance
# ---------------------------------------------------------------------------
@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    _db = get_db()

    if status is not None and status not in ALL_STATUSES:
        raise HTTPException(422, detail=f"status must be one of {sorted(ALL_STATUSES)}")

    if date_from is not None and date_to is not None:
        if date_from > date_to:
            raise HTTPException(422, detail="date_from must not be after date_to")

    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = _db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    docs = list(
        _db.attendance_logs.find(q, {"_id": 0})
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )

    return {
        "items": [serialize_attendance(doc) for doc in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
# PATCH /attendance/{emp_code}/{date}  — Regularize
# ---------------------------------------------------------------------------
@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str = Path(...),
    date: str = Path(..., description="YYYY-MM-DD"),
    body: RegularizeRequest = ...,
):
    _db = get_db()

    # Validate date format
    _validate_date(date)

    emp = _db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    record = _db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(404, detail="No attendance record for this date")

    # Determine the new values
    new_status = body.status if body.status is not None else record["status"]
    new_punch_in_dt = _ensure_aware(record.get("punch_in")) if record.get("punch_in") else None
    new_punch_out_dt = _ensure_aware(record.get("punch_out")) if record.get("punch_out") else None

    # If setting ABSENT or LEAVE, clear punch times
    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                422, detail="Cannot supply punch_in or punch_out with ABSENT/LEAVE status"
            )
        new_punch_in_dt = None
        new_punch_out_dt = None
    else:
        # Presence status
        if body.punch_in is not None:
            new_punch_in_dt = epoch_ms_to_datetime(body.punch_in)
        if body.punch_out is not None:
            new_punch_out_dt = epoch_ms_to_datetime(body.punch_out)

        # Presence status requires punch_in
        if new_punch_in_dt is None:
            raise HTTPException(422, detail="A presence status requires a punch_in")

        # punch_in must stay on the record's attendance date (R1)
        punch_in_att_date = compute_attendance_date(
            new_punch_in_dt, emp["shift_start"], emp["shift_end"]
        )
        if punch_in_att_date != date:
            raise HTTPException(
                422, detail="punch_in must belong to the record's attendance date"
            )

        # punch_out validations
        if new_punch_out_dt is not None:
            if new_punch_out_dt <= new_punch_in_dt:
                raise HTTPException(422, detail="punch_out must be after punch_in")
            if (new_punch_out_dt - new_punch_in_dt).total_seconds() > 86400:
                raise HTTPException(422, detail="punch_out must be within 24 hours of punch_in")

    # Compute derived fields for the new state
    if new_status in ("ABSENT", "LEAVE"):
        new_late_min = 0
        new_work_hours = None
        new_overtime = 0
        new_half_day = False
    else:
        new_late_min = compute_late_minutes(new_punch_in_dt, emp["shift_start"], date)
        if new_punch_out_dt is not None:
            new_work_hours = compute_work_hours(new_punch_in_dt, new_punch_out_dt)
            new_overtime = compute_overtime(
                new_punch_out_dt, date, emp["shift_start"], emp["shift_end"]
            )
            new_half_day = compute_half_day(new_work_hours)
        else:
            new_work_hours = None
            new_overtime = 0
            new_half_day = False

    # Build the changes dict (only fields that actually changed)
    old_punch_in = _ensure_aware(record.get("punch_in")) if record.get("punch_in") else None
    old_punch_out = _ensure_aware(record.get("punch_out")) if record.get("punch_out") else None
    old_status = record["status"]
    old_work_hours = record.get("work_hours")
    old_late_min = record.get("late_minutes", 0)
    old_overtime = record.get("overtime_minutes", 0)
    old_half_day = record.get("half_day", False)

    changes = {}

    if new_status != old_status:
        changes["status"] = {"from": old_status, "to": new_status}
    if new_punch_in_dt != old_punch_in:
        changes["punch_in"] = {"from": old_punch_in, "to": new_punch_in_dt}
    if new_punch_out_dt != old_punch_out:
        changes["punch_out"] = {"from": old_punch_out, "to": new_punch_out_dt}
    if new_work_hours != old_work_hours:
        changes["work_hours"] = {"from": old_work_hours, "to": new_work_hours}
    if new_late_min != old_late_min:
        changes["late_minutes"] = {"from": old_late_min, "to": new_late_min}
    if new_overtime != old_overtime:
        changes["overtime_minutes"] = {"from": old_overtime, "to": new_overtime}
    if new_half_day != old_half_day:
        changes["half_day"] = {"from": old_half_day, "to": new_half_day}

    # A request that changes nothing is 422
    if not changes:
        raise HTTPException(422, detail="No fields were changed")

    # Build history entry
    history_entry = {
        "at": datetime.now(tz=timezone.utc),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Atomic update with version check to prevent lost history entries
    old_history_len = len(record.get("history") or [])
    result = _db.attendance_logs.update_one(
        {
            "_id": record["_id"],
            "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, old_history_len]},
        },
        {
            "$set": {
                "status": new_status,
                "punch_in": new_punch_in_dt,
                "punch_out": new_punch_out_dt,
                "work_hours": new_work_hours,
                "late_minutes": new_late_min,
                "overtime_minutes": new_overtime,
                "half_day": new_half_day,
            },
            "$push": {"history": history_entry},
        },
    )

    if result.modified_count == 0:
        raise HTTPException(409, detail="Concurrent modification detected")

    updated = _db.attendance_logs.find_one({"_id": record["_id"]})
    return serialize_attendance(updated)


# ---------------------------------------------------------------------------
# GET /analytics/employees/{emp_code}/monthly
# ---------------------------------------------------------------------------
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    _db = get_db()

    emp = _db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, detail="Employee not found")

    first_day_str, last_day_str = _validate_month(month)
    first_day = date.fromisoformat(first_day_str)
    last_day = date.fromisoformat(last_day_str)

    # R7: Working days = Mon-Fri from joined_on (if mid-month)
    joined = date.fromisoformat(emp["joined_on"])
    count_from = max(first_day, joined)

    working_days = 0
    d = count_from
    while d <= last_day:
        if d.weekday() < 5:  # Mon-Fri
            working_days += 1
        d += timedelta(days=1)

    # Aggregation pipeline
    pipeline = [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first_day_str, "$lte": last_day_str}}},
        {
            "$group": {
                "_id": None,
                "present_days_full": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, False]},
                                    # Only Mon-Fri for present_days: dayOfWeek 1=Sun, 7=Sat
                                    {
                                        "$in": [
                                            {
                                                "$dayOfWeek": {
                                                    "$dateFromString": {
                                                        "dateString": "$date",
                                                        "format": "%Y-%m-%d",
                                                    }
                                                }
                                            },
                                            [2, 3, 4, 5, 6],  # Mon-Fri
                                        ]
                                    },
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "present_days_half": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                                    {
                                        "$in": [
                                            {
                                                "$dayOfWeek": {
                                                    "$dateFromString": {
                                                        "dateString": "$date",
                                                        "format": "%Y-%m-%d",
                                                    }
                                                }
                                            },
                                            [2, 3, 4, 5, 6],
                                        ]
                                    },
                                ]
                            },
                            0.5,
                            0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "total_overtime_minutes": {
                    "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                },
            }
        },
    ]

    results = list(_db.attendance_logs.aggregate(pipeline))

    if results:
        r = results[0]
        present_days = r["present_days_full"] + r["present_days_half"]
        leave_days = r["leave_days"]
        late_count = r["late_count"]
        total_late = r["total_late_minutes"]
        total_ot = r["total_overtime_minutes"]
    else:
        present_days = 0
        leave_days = 0
        late_count = 0
        total_late = 0
        total_ot = 0

    if working_days > 0:
        attendance_pct = round_half_up(present_days / working_days * 100, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_ot,
        "attendance_pct": attendance_pct,
    }


# ---------------------------------------------------------------------------
# GET /analytics/departments/summary
# ---------------------------------------------------------------------------
@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    _db = get_db()
    first_day_str, last_day_str = _validate_month(month)
    first_day = date.fromisoformat(first_day_str)
    last_day = date.fromisoformat(last_day_str)

    # Step 1: Get headcount per department (R9: employees joined on or before end of month)
    emp_filter: dict = {"joined_on": {"$lte": last_day_str}}
    if department:
        emp_filter["department"] = department

    emp_pipeline = [
        {"$match": emp_filter},
        {"$group": {"_id": "$department", "headcount": {"$sum": 1}, "emp_codes": {"$push": "$emp_code"}}},
        {"$sort": {"_id": 1}},
    ]
    dept_data = {r["_id"]: r for r in _db.employees.aggregate(emp_pipeline)}

    if not dept_data:
        return {"month": month, "items": []}

    # Step 2: Get attendance stats per department via aggregation
    att_match: dict = {"date": {"$gte": first_day_str, "$lte": last_day_str}}
    if department:
        # Only get logs from employees in the specified department
        emp_codes = dept_data.get(department, {}).get("emp_codes", [])
        att_match["emp_code"] = {"$in": emp_codes}

    att_pipeline = [
        {"$match": att_match},
        # Lookup department from employees
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp_info",
            }
        },
        {"$unwind": {"path": "$emp_info", "preserveNullAndEmptyArrays": False}},
        {
            "$group": {
                "_id": "$emp_info.department",
                "present_days_full": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, False]},
                                    {
                                        "$in": [
                                            {
                                                "$dayOfWeek": {
                                                    "$dateFromString": {
                                                        "dateString": "$date",
                                                        "format": "%Y-%m-%d",
                                                    }
                                                }
                                            },
                                            [2, 3, 4, 5, 6],
                                        ]
                                    },
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "present_days_half": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                                    {
                                        "$in": [
                                            {
                                                "$dayOfWeek": {
                                                    "$dateFromString": {
                                                        "dateString": "$date",
                                                        "format": "%Y-%m-%d",
                                                    }
                                                }
                                            },
                                            [2, 3, 4, 5, 6],
                                        ]
                                    },
                                ]
                            },
                            0.5,
                            0,
                        ]
                    }
                },
                # avg_work_hours: over present-status records that have work_hours
                "total_work_hours": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},
                                ]
                            },
                            "$work_hours",
                            0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]

    att_data = {r["_id"]: r for r in _db.attendance_logs.aggregate(att_pipeline)}

    items = []
    for dept_name in sorted(dept_data.keys()):
        dept = dept_data[dept_name]
        hc = dept["headcount"]
        if hc == 0:
            continue

        att = att_data.get(dept_name, {})
        present_days = att.get("present_days_full", 0) + att.get("present_days_half", 0)
        wh_count = att.get("work_hours_count", 0)
        avg_wh = round_half_up(att["total_work_hours"] / wh_count, 2) if wh_count > 0 else None

        items.append({
            "department": dept_name,
            "headcount": hc,
            "present_days": present_days,
            "avg_work_hours": avg_wh,
            "late_count": att.get("late_count", 0),
            "total_late_minutes": att.get("total_late_minutes", 0),
            "leave_count": att.get("leave_count", 0),
            "on_duty_count": att.get("on_duty_count", 0),
        })

    return {"month": month, "items": items}


# ---------------------------------------------------------------------------
# GET /analytics/leaderboard/late
# ---------------------------------------------------------------------------
@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = None,
):
    _db = get_db()
    first_day_str, last_day_str = _validate_month(month)

    match_stage: dict = {
        "date": {"$gte": first_day_str, "$lte": last_day_str},
    }

    pipeline: list[dict] = [
        {"$match": match_stage},
        # Join with employees to get name and department
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp_info",
            }
        },
        {"$unwind": {"path": "$emp_info", "preserveNullAndEmptyArrays": False}},
    ]

    # Filter by department if specified
    if department:
        pipeline.append({"$match": {"emp_info.department": department}})

    pipeline.extend([
        # Group by employee, sum late_minutes
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
                "name": {"$first": "$emp_info.name"},
                "department": {"$first": "$emp_info.department"},
            }
        },
        # Only employees with total > 0
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        # Sort by total_late_minutes desc, emp_code asc
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        # Use $setWindowFields for competition ranking
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {
                        "$rank": {},
                    }
                },
            }
        },
        # Filter by rank <= limit
        {"$match": {"rank": {"$lte": limit}}},
        # Project final shape
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": 1,
                "department": 1,
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ])

    items = list(_db.attendance_logs.aggregate(pipeline))

    return {"month": month, "items": items}


# ---------------------------------------------------------------------------
# GET /analytics/departments/{department}/trend
# ---------------------------------------------------------------------------
@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str = Path(...),
    date_from: str = Query(..., alias="from"),
    date_to: str = Query(..., alias="to"),
):
    _db = get_db()

    from_date = _validate_date(date_from)
    to_date = _validate_date(date_to)

    if to_date < from_date:
        raise HTTPException(422, detail="'to' must not be before 'from'")
    if (to_date - from_date).days > 92:
        raise HTTPException(422, detail="Date range must not exceed 92 days")

    # Check department exists
    dept_exists = _db.employees.find_one({"department": department})
    if not dept_exists:
        raise HTTPException(404, detail="Unknown department (no employees)")

    from_str = from_date.isoformat()
    to_str = to_date.isoformat()

    # Get all employees in this department with their joined_on dates
    dept_employees = list(
        _db.employees.find(
            {"department": department},
            {"_id": 0, "emp_code": 1, "joined_on": 1},
        )
    )

    # Build day-by-day results — the contract says to use $densify,
    # but we need to compute headcount per day anyway, so we do a hybrid approach:
    # 1. Aggregate attendance data from MongoDB
    # 2. Fill gaps in Python with precomputed headcounts

    # Get attendance data grouped by date
    att_pipeline = [
        {
            "$match": {
                "date": {"$gte": from_str, "$lte": to_str},
                "emp_code": {"$in": [e["emp_code"] for e in dept_employees]},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count_full": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, False]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "present_count_half": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", list(PRESENCE_STATUSES)]},
                                    {"$eq": [{"$ifNull": ["$half_day", False]}, True]},
                                ]
                            },
                            0.5,
                            0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {
                        "$cond": [
                            {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                            1,
                            0,
                        ]
                    }
                },
            }
        },
    ]
    att_by_date = {
        r["_id"]: r for r in _db.attendance_logs.aggregate(att_pipeline)
    }

    # Build daily items
    items = []
    rates_window: list[Optional[float]] = []  # sliding window for moving_avg_7d

    d = from_date
    while d <= to_date:
        d_str = d.isoformat()
        is_working = d.weekday() < 5  # Mon-Fri

        # Headcount: employees with joined_on <= this day (R9)
        hc = sum(1 for e in dept_employees if e["joined_on"] <= d_str)

        att = att_by_date.get(d_str, {})
        present_count = att.get("present_count_full", 0) + att.get("present_count_half", 0)
        late_count = att.get("late_count", 0)

        # attendance_rate: only on working days with headcount > 0
        if is_working and hc > 0:
            rate = round_half_up(present_count / hc, 4)
        else:
            rate = None

        rates_window.append(rate)

        # moving_avg_7d: mean of non-null rates in the last 7 rows (within requested range)
        window = rates_window[-7:]
        non_null = [r for r in window if r is not None]
        if non_null:
            moving_avg = round_half_up(sum(non_null) / len(non_null), 4)
        else:
            moving_avg = None

        items.append({
            "date": d_str,
            "is_working_day": is_working,
            "headcount": hc,
            "present_count": present_count,
            "late_count": late_count,
            "attendance_rate": rate,
            "moving_avg_7d": moving_avg,
        })

        d += timedelta(days=1)

    return {"department": department, "items": items}


# ---------------------------------------------------------------------------
# GET /admin/explain/{endpoint}
# ---------------------------------------------------------------------------
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str = Path(
        ...,
        description="One of: attendance_list, employee_monthly, department_summary, late_leaderboard, department_trend",
    ),
    # All possible query params from the explained endpoints
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    date_from_alias: Optional[str] = Query(default=None, alias="from"),
    date_to_alias: Optional[str] = Query(default=None, alias="to"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    _db = get_db()

    valid_endpoints = {
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    }
    if endpoint not in valid_endpoints:
        raise HTTPException(422, detail=f"endpoint must be one of {sorted(valid_endpoints)}")

    if endpoint == "attendance_list":
        q: dict = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status

        skip = (page - 1) * page_size
        cmd = {
            "explain": {
                "find": "attendance_logs",
                "filter": q,
                "sort": {"date": -1, "emp_code": 1},
                "skip": skip,
                "limit": page_size,
            },
            "verbosity": "executionStats",
        }
        explain = _db.command(cmd)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": _clean_bson(explain)}

    elif endpoint == "employee_monthly":
        if not emp_code:
            raise HTTPException(422, detail="emp_code is required for employee_monthly")
        if not month:
            raise HTTPException(422, detail="month is required for employee_monthly")

        first_day_str, last_day_str = _validate_month(month)
        pipeline = [
            {"$match": {"emp_code": emp_code, "date": {"$gte": first_day_str, "$lte": last_day_str}}},
            {"$group": {"_id": None, "count": {"$sum": 1}}},
        ]
        cmd = {
            "explain": {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            "verbosity": "executionStats",
        }
        explain = _db.command(cmd)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": _clean_bson(explain)}

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(422, detail="month is required for department_summary")
        first_day_str, last_day_str = _validate_month(month)

        match_stage: dict = {"date": {"$gte": first_day_str, "$lte": last_day_str}}
        pipeline = [
            {"$match": match_stage},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "emp_code",
                    "foreignField": "emp_code",
                    "as": "emp_info",
                }
            },
            {"$unwind": "$emp_info"},
            {"$group": {"_id": "$emp_info.department", "count": {"$sum": 1}}},
        ]
        if department:
            pipeline.insert(1, {"$match": {"emp_info.department": department}})

        cmd = {
            "explain": {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            "verbosity": "executionStats",
        }
        explain = _db.command(cmd)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": _clean_bson(explain)}

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(422, detail="month is required for late_leaderboard")
        first_day_str, last_day_str = _validate_month(month)

        match_stage = {"date": {"$gte": first_day_str, "$lte": last_day_str}}
        pipeline: list = [
            {"$match": match_stage},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "emp_code",
                    "foreignField": "emp_code",
                    "as": "emp_info",
                }
            },
            {"$unwind": "$emp_info"},
        ]
        if department:
            pipeline.append({"$match": {"emp_info.department": department}})
        pipeline.extend([
            {
                "$group": {
                    "_id": "$emp_code",
                    "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                }
            },
            {"$match": {"total_late_minutes": {"$gt": 0}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
        ])

        cmd = {
            "explain": {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            "verbosity": "executionStats",
        }
        explain = _db.command(cmd)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": _clean_bson(explain)}

    elif endpoint == "department_trend":
        if not department:
            raise HTTPException(422, detail="department is required for department_trend")
        f = date_from_alias
        t = date_to_alias
        if not f or not t:
            raise HTTPException(422, detail="from and to are required for department_trend")

        # Get emp_codes in department
        emp_codes = [
            e["emp_code"]
            for e in _db.employees.find({"department": department}, {"emp_code": 1, "_id": 0})
        ]

        pipeline = [
            {
                "$match": {
                    "date": {"$gte": f, "$lte": t},
                    "emp_code": {"$in": emp_codes},
                }
            },
            {"$group": {"_id": "$date", "count": {"$sum": 1}}},
        ]

        cmd = {
            "explain": {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            "verbosity": "executionStats",
        }
        explain = _db.command(cmd)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": _clean_bson(explain)}

    raise HTTPException(422, detail="Unknown endpoint")
