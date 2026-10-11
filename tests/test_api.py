"""
Tests for the Employee Attendance & Analytics API.

These tests use FastAPI's TestClient with a real or mock MongoDB instance.
Set MONGO_URI and MONGO_DB environment variables to run against a real database.
If MongoDB is unavailable, tests that require it will be skipped.
"""
import os
import time
import math
import concurrent.futures
import pytest
from datetime import datetime, timezone, timedelta, date
from unittest.mock import patch
from decimal import Decimal, ROUND_HALF_UP

# Set test DB before importing app
os.environ.setdefault("MONGO_DB", "attendance_test_db")

from fastapi.testclient import TestClient
from app.main import (
    app, epoch_ms_to_datetime, datetime_to_epoch_ms,
    compute_attendance_date, compute_late_minutes, compute_work_hours,
    compute_overtime, compute_half_day, round_half_up,
    IST,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mongo_available():
    """Check if MongoDB is reachable."""
    try:
        from pymongo import MongoClient
        c = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=2000)
        c.admin.command("ping")
        c.close()
        return True
    except Exception:
        return False


MONGO_OK = _mongo_available()
skip_no_mongo = pytest.mark.skipif(not MONGO_OK, reason="MongoDB not available")


@pytest.fixture(scope="module")
def client():
    """Create a test client. Seeds test data on setup, cleans on teardown."""
    if not MONGO_OK:
        pytest.skip("MongoDB not available")
    with TestClient(app) as c:
        # Clean test data
        from pymongo import MongoClient as MC
        mongo = MC(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        test_db = mongo[os.getenv("MONGO_DB", "attendance_test_db")]
        test_db.employees.delete_many({})
        test_db.attendance_logs.delete_many({})
        mongo.close()
        yield c
        # Cleanup
        mongo = MC(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        test_db = mongo[os.getenv("MONGO_DB", "attendance_test_db")]
        test_db.employees.delete_many({})
        test_db.attendance_logs.delete_many({})
        mongo.close()


def _find_explain_stages(obj, target):
    """Recursively find all stages matching target in explain plan."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "stage" and v == target:
                found.append(obj)
            else:
                found.extend(_find_explain_stages(v, target))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_explain_stages(item, target))
    return found


# ---------------------------------------------------------------------------
# Unit tests for helpers (no DB required)
# ---------------------------------------------------------------------------

class TestRoundHalfUp:
    def test_basic(self):
        assert round_half_up(9.125, 2) == 9.13   # half-up, not banker's
        assert round_half_up(9.115, 2) == 9.12    # exact half rounds up
        assert round_half_up(2.5, 0) == 3.0
        assert round_half_up(3.5, 0) == 4.0       # banker's would give 4 too, but 2.5 -> 2 with banker's

    def test_four_decimals(self):
        assert round_half_up(0.12345, 4) == 0.1235
        assert round_half_up(0.12344, 4) == 0.1234

    def test_none(self):
        assert round_half_up(None, 2) is None

    def test_decimal_boundary_half_up(self):
        # Averaging 6.05 and 6.10 produces 6.075 which must round to 6.08 under half-up
        avg_dec = (Decimal("6.05") + Decimal("6.10")) / Decimal("2")
        assert round_half_up(avg_dec, 2) == 6.08


class TestEpochConversion:
    def test_roundtrip(self):
        ms = 1783312500000
        dt = epoch_ms_to_datetime(ms)
        assert dt.tzinfo is not None  # UTC-aware
        result_ms = datetime_to_epoch_ms(dt)
        assert result_ms == ms

    def test_truncation(self):
        """R1: truncated to whole seconds."""
        ms = 1783312500999  # .999 fractional
        dt = epoch_ms_to_datetime(ms)
        assert dt.microsecond == 0
        assert datetime_to_epoch_ms(dt) == 1783312500000

    def test_none(self):
        assert datetime_to_epoch_ms(None) is None


class TestAttendanceDate:
    def test_normal_shift(self):
        """Normal shift: attendance date = IST date of punch_in."""
        # 2026-07-06T03:58:00Z = 2026-07-06 09:28 IST
        dt = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        result = compute_attendance_date(dt, "09:30", "18:30")
        assert result == "2026-07-06"

    def test_overnight_shift_evening_punch(self):
        """Overnight shift: punch-in in the evening → same day."""
        # 2026-07-06 21:55 IST = 2026-07-06T16:25:00Z
        dt = datetime(2026, 7, 6, 16, 25, 0, tzinfo=timezone.utc)
        result = compute_attendance_date(dt, "22:00", "06:00")
        assert result == "2026-07-06"

    def test_overnight_shift_early_morning_punch(self):
        """Overnight shift: punch-in before shift_end → belongs to previous day."""
        # 2026-07-07 02:00 IST = 2026-07-06T20:30:00Z
        dt = datetime(2026, 7, 6, 20, 30, 0, tzinfo=timezone.utc)
        result = compute_attendance_date(dt, "22:00", "06:00")
        assert result == "2026-07-06"


class TestLateMinutes:
    def test_within_grace(self):
        """R2: 09:40:00 with shift_start 09:30 → 10 min exactly → 0 (grace inclusive)."""
        dt = datetime(2026, 7, 6, 4, 10, 0, tzinfo=timezone.utc)  # 09:40 IST
        result = compute_late_minutes(dt, "09:30", "2026-07-06")
        assert result == 0

    def test_one_second_late(self):
        """R2: 09:40:01 → 10 minutes (floor of 601/60 = 10.01 → 10)."""
        dt = datetime(2026, 7, 6, 4, 10, 1, tzinfo=timezone.utc)  # 09:40:01 IST
        result = compute_late_minutes(dt, "09:30", "2026-07-06")
        assert result == 10

    def test_35_minutes_late(self):
        """R2: 10:05 with shift 09:30 → 35 minutes late."""
        dt = datetime(2026, 7, 6, 4, 35, 0, tzinfo=timezone.utc)  # 10:05 IST
        result = compute_late_minutes(dt, "09:30", "2026-07-06")
        assert result == 35

    def test_45_59_late(self):
        """R2: 10:15:59 → floor((45*60+59)/60) = floor(45.98) = 45."""
        dt = datetime(2026, 7, 6, 4, 45, 59, tzinfo=timezone.utc)  # 10:15:59 IST
        result = compute_late_minutes(dt, "09:30", "2026-07-06")
        assert result == 45


class TestWorkHours:
    def test_basic(self):
        """R4: (punch_out - punch_in) in seconds / 3600, 2 decimals half-up."""
        pin = datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc)
        pout = datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc)
        wh = compute_work_hours(pin, pout)
        # 9 hours 7 minutes = 547 minutes = 32820 seconds / 3600 = 9.1166... → 9.12
        assert wh == 9.12

    def test_half_day(self):
        """3.5 hours → half_day = true."""
        pin = datetime(2026, 7, 7, 4, 0, 0, tzinfo=timezone.utc)
        pout = datetime(2026, 7, 7, 7, 30, 0, tzinfo=timezone.utc)
        wh = compute_work_hours(pin, pout)
        assert wh == 3.5
        assert compute_half_day(wh) is True

    def test_full_day(self):
        assert compute_half_day(9.12) is False
        assert compute_half_day(4.50) is False  # exactly 4.50 is NOT half day

    def test_boundary(self):
        assert compute_half_day(4.49) is True   # < 4.50
        assert compute_half_day(4.50) is False  # >= 4.50


class TestOvertime:
    def test_under_30(self):
        """R3: < 30 minutes after shift_end → 0."""
        pout = datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc)  # 18:35 IST, 5 min after 18:30
        result = compute_overtime(pout, "2026-07-06", "09:30", "18:30")
        assert result == 0

    def test_40_minutes(self):
        """R3: 40 minutes after shift_end → 40."""
        pout = datetime(2026, 7, 6, 13, 40, 0, tzinfo=timezone.utc)  # 19:10 IST
        result = compute_overtime(pout, "2026-07-06", "09:30", "18:30")
        assert result == 40

    def test_overnight_shift(self):
        """Overnight shift: shift_end 06:00 next day."""
        # punch_out 2026-07-07 06:40 IST = 2026-07-07T01:10:00Z
        pout = datetime(2026, 7, 7, 1, 10, 0, tzinfo=timezone.utc)
        result = compute_overtime(pout, "2026-07-06", "22:00", "06:00")
        assert result == 40


# ---------------------------------------------------------------------------
# Integration tests (require MongoDB)
# ---------------------------------------------------------------------------

@skip_no_mongo
class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_health_timeout_when_mongo_down(self, client):
        """When MongoDB is unreachable, /health must return 503 promptly within <= 3.5s."""
        from unittest.mock import MagicMock
        mock_db = MagicMock()
        mock_db.command.side_effect = Exception("MongoDB connection refused")
        with patch("app.main.get_db", return_value=mock_db):
            start = time.perf_counter()
            r = client.get("/health")
            elapsed = time.perf_counter() - start
            assert r.status_code == 503
            assert elapsed < 3.5

    def test_health_short_server_selection_timeout(self, client):
        """Verify MongoClient serverSelectionTimeoutMS configuration is 3000ms (3.0s)."""
        import app.main as app_main
        assert app_main.client is not None
        assert app_main.client.options.server_selection_timeout == 3.0


@skip_no_mongo
class TestCreateEmployee:
    def test_create_success(self, client):
        r = client.post("/employees", json={
            "emp_code": "EMP0001",
            "name": "Asha Rao",
            "email": "asha@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-05",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["emp_code"] == "EMP0001"
        assert data["name"] == "Asha Rao"
        assert isinstance(data["created_at"], int)  # epoch ms
        assert data["created_at"] >= 100000000000

    def test_duplicate_409(self, client):
        r = client.post("/employees", json={
            "emp_code": "EMP0001",
            "name": "Duplicate",
            "email": "dup@example.com",
            "department": "Engineering",
            "joined_on": "2026-01-05",
        })
        assert r.status_code == 409

    def test_invalid_emp_code(self, client):
        r = client.post("/employees", json={
            "emp_code": "INVALID",
            "name": "Test",
            "email": "t@t.com",
            "department": "Eng",
            "joined_on": "2026-01-01",
        })
        assert r.status_code == 422

    def test_same_shift(self, client):
        r = client.post("/employees", json={
            "emp_code": "EMP9999",
            "name": "Test",
            "email": "t@t.com",
            "department": "Eng",
            "shift_start": "09:00",
            "shift_end": "09:00",
            "joined_on": "2026-01-01",
        })
        assert r.status_code == 422

    def test_create_more_employees(self, client):
        """Seed additional employees for later tests."""
        employees = [
            {"emp_code": "EMP0002", "name": "Vikram Shah", "email": "vikram@example.com", "department": "Engineering", "shift_start": "09:30", "shift_end": "18:30", "joined_on": "2026-07-13"},
            {"emp_code": "EMP0003", "name": "Neha Iyer", "email": "neha@example.com", "department": "Sales", "shift_start": "09:30", "shift_end": "18:30", "joined_on": "2026-02-01"},
            {"emp_code": "EMP0004", "name": "Rohit Das", "email": "rohit@example.com", "department": "Sales", "shift_start": "09:30", "shift_end": "18:30", "joined_on": "2026-02-01"},
            {"emp_code": "EMP0005", "name": "Meera Nair", "email": "meera@example.com", "department": "Support", "shift_start": "22:00", "shift_end": "06:00", "joined_on": "2026-03-10"},
            {"emp_code": "EMP0006", "name": "Karan Bedi", "email": "karan@example.com", "department": "Support", "shift_start": "09:30", "shift_end": "18:30", "joined_on": "2026-03-10"},
        ]
        for emp in employees:
            r = client.post("/employees", json=emp)
            assert r.status_code == 201, f"Failed to create {emp['emp_code']}: {r.json()}"


@skip_no_mongo
class TestListEmployees:
    def test_list_all(self, client):
        r = client.get("/employees")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 6
        # Verify sorted by emp_code ascending
        codes = [e["emp_code"] for e in data["items"]]
        assert codes == sorted(codes)

    def test_list_filtered(self, client):
        r = client.get("/employees", params={"department": "Engineering"})
        assert r.status_code == 200
        data = r.json()
        for item in data["items"]:
            assert item["department"] == "Engineering"
        assert data["total"] == len(data["items"])  # total matches filter

    def test_pagination(self, client):
        r = client.get("/employees", params={"page": 1, "page_size": 2})
        assert r.status_code == 200
        data = r.json()
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2


@skip_no_mongo
class TestPunchIn:
    def test_punch_in_success(self, client):
        # 2026-07-06 09:28 IST = 03:58 UTC
        ts = int(datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0001",
            "punched_at": ts,
            "status": "PRESENT",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["emp_code"] == "EMP0001"
        assert data["date"] == "2026-07-06"
        assert data["status"] == "PRESENT"
        assert data["punch_in"] == ts
        assert data["punch_out"] is None
        assert data["work_hours"] is None
        assert data["late_minutes"] == 0
        assert data["overtime_minutes"] == 0
        assert data["half_day"] is False
        assert data["history"] == []

    def test_duplicate_punch_in_409(self, client):
        ts = int(datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0001",
            "punched_at": ts,
        })
        assert r.status_code == 409

    def test_unknown_employee_404(self, client):
        ts = int(datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP9999",
            "punched_at": ts,
        })
        assert r.status_code == 404

    def test_invalid_status_422(self, client):
        ts = int(datetime(2026, 7, 6, 3, 58, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0002",
            "punched_at": ts,
            "status": "ABSENT",
        })
        assert r.status_code == 422

    def test_invalid_epoch_ms(self, client):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0002",
            "punched_at": 12345,  # below min
        })
        assert r.status_code == 422

    def test_late_punch_in(self, client):
        # EMP0003 shift 09:30. Punch in at 10:05 IST = 04:35 UTC
        ts = int(datetime(2026, 7, 6, 4, 35, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0003",
            "punched_at": ts,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["late_minutes"] == 35

    def test_overnight_shift_punch_in(self, client):
        # EMP0005 shift 22:00-06:00. Punch in at 21:55 IST = 16:25 UTC
        ts = int(datetime(2026, 7, 6, 16, 25, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0005",
            "punched_at": ts,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["date"] == "2026-07-06"
        assert data["late_minutes"] == 0


@skip_no_mongo
class TestPunchOut:
    def test_punch_out_success(self, client):
        # EMP0001 punched in at 2026-07-06 03:58:00 UTC. Punch out at 13:05 UTC (18:35 IST)
        ts = int(datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-out", json={
            "emp_code": "EMP0001",
            "punched_at": ts,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["work_hours"] == 9.12
        assert data["overtime_minutes"] == 0  # 5 min after shift end, < 30
        assert data["half_day"] is False

    def test_punch_out_with_overtime(self, client):
        # EMP0003: punch out at 19:10 IST = 13:40 UTC
        ts = int(datetime(2026, 7, 6, 13, 40, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-out", json={
            "emp_code": "EMP0003",
            "punched_at": ts,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["overtime_minutes"] == 40

    def test_punch_out_overnight(self, client):
        # EMP0005: punch out at 06:40 IST next day = 2026-07-07T01:10:00Z
        ts = int(datetime(2026, 7, 7, 1, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-out", json={
            "emp_code": "EMP0005",
            "punched_at": ts,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["overtime_minutes"] == 40

    def test_no_open_session_404(self, client):
        ts = int(datetime(2026, 7, 6, 13, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-out", json={
            "emp_code": "EMP0002",  # never punched in
            "punched_at": ts,
        })
        assert r.status_code == 404

    def test_already_punched_out_409(self, client):
        # EMP0001 already punched out -> 409 Conflict per openapi contract
        ts = int(datetime(2026, 7, 6, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-out", json={
            "emp_code": "EMP0001",
            "punched_at": ts,
        })
        assert r.status_code == 409

    def test_punch_out_before_in_422(self, client):
        # First create a new punch-in for EMP0004
        ts_in = int(datetime(2026, 7, 7, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0004",
            "punched_at": ts_in,
            "status": "WFH",
        })
        assert r.status_code == 201

        # Try to punch out before punch in
        ts_out = int(datetime(2026, 7, 7, 3, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.post("/attendance/punch-out", json={
            "emp_code": "EMP0004",
            "punched_at": ts_out,
        })
        assert r.status_code == 422  # punch_out timestamp <= punch_in is 422 per OpenAPI specification


@skip_no_mongo
class TestListAttendance:
    def test_list_all(self, client):
        r = client.get("/attendance")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data

    def test_filter_by_emp_code(self, client):
        r = client.get("/attendance", params={"emp_code": "EMP0001"})
        assert r.status_code == 200
        for item in r.json()["items"]:
            assert item["emp_code"] == "EMP0001"

    def test_date_range(self, client):
        r = client.get("/attendance", params={"date_from": "2026-07-06", "date_to": "2026-07-06"})
        assert r.status_code == 200
        for item in r.json()["items"]:
            assert item["date"] == "2026-07-06"

    def test_invalid_date_range(self, client):
        r = client.get("/attendance", params={"date_from": "2026-07-07", "date_to": "2026-07-06"})
        assert r.status_code == 422

    def test_invalid_calendar_date(self, client):
        # Reject invalid calendar dates like Feb 31 or month 13
        r = client.get("/attendance", params={"date_from": "2026-02-31"})
        assert r.status_code == 422
        r = client.get("/attendance", params={"date_to": "2026-13-01"})
        assert r.status_code == 422
        r = client.get("/attendance", params={"date_from": "invalid-format"})
        assert r.status_code == 422

    def test_invalid_status_filter(self, client):
        r = client.get("/attendance", params={"status": "INVALID_STATUS"})
        assert r.status_code == 422


@skip_no_mongo
class TestRegularize:
    def test_regularize_success(self, client):
        # First punch out EMP0004 so we have a complete record
        ts_out = int(datetime(2026, 7, 7, 7, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
        client.post("/attendance/punch-out", json={"emp_code": "EMP0004", "punched_at": ts_out})

        # Regularize: change punch_in
        new_punch_in = int(datetime(2026, 7, 7, 3, 58, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.patch("/attendance/EMP0004/2026-07-07", json={
            "reason": "biometric glitch fix",
            "regularized_by": "hr.admin",
            "punch_in": new_punch_in,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["punch_in"] == new_punch_in
        assert len(data["history"]) == 1
        assert data["history"][0]["by"] == "hr.admin"
        assert "punch_in" in data["history"][0]["changes"]

    def test_no_change_422(self, client):
        # Re-send with same punch_in → nothing changed
        r = client.get("/attendance", params={"emp_code": "EMP0004", "date_from": "2026-07-07", "date_to": "2026-07-07"})
        current = r.json()["items"][0]
        r = client.patch("/attendance/EMP0004/2026-07-07", json={
            "reason": "test no change here",
            "regularized_by": "admin",
            "punch_in": current["punch_in"],
        })
        assert r.status_code == 422

    def test_absent_with_punch_422(self, client):
        r = client.patch("/attendance/EMP0004/2026-07-07", json={
            "reason": "trying invalid change",
            "regularized_by": "admin",
            "status": "ABSENT",
            "punch_in": 1783312500000,
        })
        assert r.status_code == 422

    def test_unknown_employee_404(self, client):
        r = client.patch("/attendance/EMP9999/2026-07-07", json={
            "reason": "test missing employee",
            "regularized_by": "admin",
            "status": "ABSENT",
        })
        assert r.status_code == 404

    def test_no_record_404(self, client):
        r = client.patch("/attendance/EMP0001/2025-01-01", json={
            "reason": "test missing record",
            "regularized_by": "admin",
            "status": "ABSENT",
        })
        assert r.status_code == 404

    def test_history_preserved(self, client):
        """Multiple regularizations should stack history entries."""
        new_punch_in2 = int(datetime(2026, 7, 7, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r = client.patch("/attendance/EMP0004/2026-07-07", json={
            "reason": "second correction here",
            "regularized_by": "manager",
            "punch_in": new_punch_in2,
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["history"]) == 2
        assert data["history"][0]["by"] == "hr.admin"
        assert data["history"][1]["by"] == "manager"


@skip_no_mongo
class TestEmployeeMonthly:
    def test_basic(self, client):
        r = client.get("/analytics/employees/EMP0001/monthly", params={"month": "2026-07"})
        assert r.status_code == 200
        data = r.json()
        assert data["emp_code"] == "EMP0001"
        assert data["month"] == "2026-07"
        assert "working_days" in data
        assert "present_days" in data
        assert "attendance_pct" in data

    def test_unknown_employee(self, client):
        r = client.get("/analytics/employees/EMP9999/monthly", params={"month": "2026-07"})
        assert r.status_code == 404

    def test_invalid_month(self, client):
        r = client.get("/analytics/employees/EMP0001/monthly", params={"month": "2026-13"})
        assert r.status_code == 422


@skip_no_mongo
class TestDepartmentSummary:
    def test_basic(self, client):
        r = client.get("/analytics/departments/summary", params={"month": "2026-07"})
        assert r.status_code == 200
        data = r.json()
        assert data["month"] == "2026-07"
        assert "items" in data
        for item in data["items"]:
            assert "department" in item
            assert "headcount" in item
            assert item["headcount"] > 0

    def test_filter_dept(self, client):
        r = client.get("/analytics/departments/summary", params={"month": "2026-07", "department": "Engineering"})
        assert r.status_code == 200
        data = r.json()
        for item in data["items"]:
            assert item["department"] == "Engineering"

    def test_avg_work_hours_half_boundary_rounding(self, client):
        # Insert 2 employees with work_hours 6.05 and 6.10 to verify Decimal half-up boundary rounding
        from pymongo import MongoClient
        mongo = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        test_db = mongo[os.getenv("MONGO_DB", "attendance_test_db")]
        test_db.employees.insert_many([
            {
                "emp_code": "EMP8881",
                "name": "Round Test 1",
                "email": "rt1@example.com",
                "department": "PrecisionTest",
                "shift_start": "09:00",
                "shift_end": "18:00",
                "joined_on": "2026-01-01",
                "created_at": datetime.now(tz=timezone.utc),
            },
            {
                "emp_code": "EMP8882",
                "name": "Round Test 2",
                "email": "rt2@example.com",
                "department": "PrecisionTest",
                "shift_start": "09:00",
                "shift_end": "18:00",
                "joined_on": "2026-01-01",
                "created_at": datetime.now(tz=timezone.utc),
            },
        ])
        test_db.attendance_logs.insert_many([
            {
                "emp_code": "EMP8881",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 6, 3, 30, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 6, 9, 33, tzinfo=timezone.utc),
                "work_hours": 6.05,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
            },
            {
                "emp_code": "EMP8882",
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": datetime(2026, 7, 6, 3, 30, tzinfo=timezone.utc),
                "punch_out": datetime(2026, 7, 6, 9, 36, tzinfo=timezone.utc),
                "work_hours": 6.10,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": False,
            },
        ])
        mongo.close()

        r = client.get("/analytics/departments/summary", params={"month": "2026-07", "department": "PrecisionTest"})
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["avg_work_hours"] == 6.08


@skip_no_mongo
class TestLateLeaderboard:
    def test_basic(self, client):
        r = client.get("/analytics/leaderboard/late", params={"month": "2026-07"})
        assert r.status_code == 200
        data = r.json()
        assert data["month"] == "2026-07"
        assert "items" in data
        for item in data["items"]:
            assert "rank" in item
            assert item["total_late_minutes"] > 0

    def test_with_limit(self, client):
        r = client.get("/analytics/leaderboard/late", params={"month": "2026-07", "limit": 1})
        assert r.status_code == 200
        data = r.json()
        for item in data["items"]:
            assert item["rank"] <= 1


@skip_no_mongo
class TestDepartmentTrend:
    def test_basic(self, client):
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-06",
            "to": "2026-07-10",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["department"] == "Engineering"
        # Should have exactly 5 days
        assert len(data["items"]) == 5
        for item in data["items"]:
            assert "date" in item
            assert "is_working_day" in item
            assert "headcount" in item
            assert "present_count" in item
            assert "attendance_rate" in item or item.get("attendance_rate") is None

    def test_unknown_dept_404(self, client):
        r = client.get("/analytics/departments/NonExistent/trend", params={
            "from": "2026-07-06",
            "to": "2026-07-10",
        })
        assert r.status_code == 404

    def test_invalid_range(self, client):
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-10",
            "to": "2026-07-06",
        })
        assert r.status_code == 422

    def test_range_too_long(self, client):
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-01-01",
            "to": "2026-12-31",
        })
        assert r.status_code == 422


@skip_no_mongo
class TestExplainEndpoint:
    def test_attendance_list(self, client):
        r = client.get("/admin/explain/attendance_list")
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "attendance_list"
        assert data["collection"] == "attendance_logs"
        assert "explain" in data

    def test_employee_monthly(self, client):
        r = client.get("/admin/explain/employee_monthly", params={"emp_code": "EMP0001", "month": "2026-07"})
        assert r.status_code == 200

    def test_missing_params(self, client):
        r = client.get("/admin/explain/employee_monthly")
        assert r.status_code == 422

    def test_department_summary(self, client):
        r = client.get("/admin/explain/department_summary", params={"month": "2026-07"})
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "department_summary"
        assert "explain" in data

    def test_late_leaderboard(self, client):
        r = client.get("/admin/explain/late_leaderboard", params={"month": "2026-07"})
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "late_leaderboard"
        assert "explain" in data

    def test_department_trend(self, client):
        r = client.get("/admin/explain/department_trend", params={
            "department": "Engineering",
            "from": "2026-07-01",
            "to": "2026-07-05",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "department_trend"
        assert "explain" in data

    def test_invalid_endpoint(self, client):
        r = client.get("/admin/explain/invalid_endpoint")
        assert r.status_code == 422


@skip_no_mongo
class TestStrictInputValidation:
    def test_punch_in_rejects_string_timestamp(self, client):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0001",
            "punched_at": "1786335300000",
        })
        assert r.status_code == 422

    def test_punch_in_rejects_float_timestamp(self, client):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0001",
            "punched_at": 1786335300000.5,
        })
        assert r.status_code == 422

    def test_punch_in_rejects_boolean_timestamp(self, client):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0001",
            "punched_at": True,
        })
        assert r.status_code == 422

    def test_punch_in_rejects_seconds_timestamp(self, client):
        r = client.post("/attendance/punch-in", json={
            "emp_code": "EMP0001",
            "punched_at": 1786335300,
        })
        assert r.status_code == 422

    def test_punch_out_rejects_non_integer_timestamp(self, client):
        for bad_val in ["1786335300000", 1786335300000.5, True, 1786335300]:
            r = client.post("/attendance/punch-out", json={
                "emp_code": "EMP0001",
                "punched_at": bad_val,
            })
            assert r.status_code == 422

    def test_regularize_rejects_non_integer_timestamp(self, client):
        r = client.patch("/attendance/EMP0001/2026-07-06", json={
            "punch_in": "1786335300000",
            "reason": "Test regularize validation",
            "regularized_by": "HR_ADMIN",
        })
        assert r.status_code == 422

    def test_employee_joined_on_rejects_invalid_calendar_date(self, client):
        r = client.post("/employees", json={
            "emp_code": "EMP8888",
            "name": "Invalid Date",
            "email": "inv@example.com",
            "department": "Engineering",
            "joined_on": "2026-02-31",
        })
        assert r.status_code == 422

    def test_employee_joined_on_rejects_malformed_pattern(self, client):
        for bad_date in ["2026-2-5", "2026/02/05", "05-02-2026", "2026-02-05T00:00:00"]:
            r = client.post("/employees", json={
                "emp_code": "EMP8889",
                "name": "Bad Format",
                "email": "bad@example.com",
                "department": "Engineering",
                "joined_on": bad_date,
            })
            assert r.status_code == 422

    def test_trend_rejects_invalid_calendar_date(self, client):
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-02-29",
            "to": "2026-03-05",
        })
        assert r.status_code == 422

    def test_trend_rejects_malformed_date(self, client):
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-7-6",
            "to": "2026-07-10",
        })
        assert r.status_code == 422


@skip_no_mongo
class TestDepartmentTrendAggregation:
    def test_gap_filling_and_moving_average(self, client):
        """
        Verify pure MongoDB aggregation:
        - Gap filling for dates with no attendance logs via $densify
        - Working days vs weekends
        - 7-row moving average via $setWindowFields
        """
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-01",
            "to": "2026-07-14",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["department"] == "Engineering"
        items = data["items"]
        assert len(items) == 14

        dates = [item["date"] for item in items]
        assert dates[0] == "2026-07-01"
        assert dates[-1] == "2026-07-14"

        # Check weekends: 2026-07-04 is Saturday, 2026-07-05 is Sunday
        sat = next(i for i in items if i["date"] == "2026-07-04")
        sun = next(i for i in items if i["date"] == "2026-07-05")
        assert sat["is_working_day"] is False
        assert sat["attendance_rate"] is None
        # Preceding rows (July 1, 2, 3) were working days with 0.0 rate, so 7d moving avg is 0.0
        assert sat["moving_avg_7d"] == 0.0

        assert sun["is_working_day"] is False
        assert sun["attendance_rate"] is None
        assert sun["moving_avg_7d"] == 0.0

        # Check working day gap
        wday = next(i for i in items if i["date"] == "2026-07-01")
        assert wday["is_working_day"] is True
        assert wday["present_count"] >= 0.0
        assert wday["headcount"] >= 1

    def test_weekend_only_trend_returns_null_moving_avg(self, client):
        """When the range starts on weekend and has no preceding non-null rates, moving_avg_7d is None."""
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-04",
            "to": "2026-07-05",
        })
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 2
        for item in items:
            assert item["is_working_day"] is False
            assert item["attendance_rate"] is None
            assert item["moving_avg_7d"] is None

    def test_trend_headcount_joined_on(self, client):
        """Verify R9 headcount: only employees with joined_on <= date are counted."""
        r = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-12",
            "to": "2026-07-14",
        })
        assert r.status_code == 200
        items = r.json()["items"]
        # EMP0001 joined 2026-01-05; EMP0002 joined 2026-07-13
        d12 = next(i for i in items if i["date"] == "2026-07-12")
        d13 = next(i for i in items if i["date"] == "2026-07-13")
        assert d13["headcount"] == d12["headcount"] + 1


@skip_no_mongo
class TestExplainFilteringAndIndexes:
    def test_department_summary_explain_with_department(self, client):
        """Verify department_summary explain works with department filtering without pipeline errors and with index scans."""
        client.post("/employees", json={
            "emp_code": "EMP0001",
            "name": "Eng Test",
            "email": "engtest@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        r = client.get("/admin/explain/department_summary", params={
            "month": "2026-07",
            "department": "Engineering",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "department_summary"
        assert "explain" in data
        exp = data["explain"]
        assert _find_explain_stages(exp, "COLLSCAN") == []
        assert len(_find_explain_stages(exp, "IXSCAN")) >= 1
        # Also verify lookup stage uses index
        lookup_stages = [s for s in exp.get("stages", []) if any("lookup" in k for k in s.keys())]
        assert len(lookup_stages) >= 1
        assert "indexesUsed" in lookup_stages[0]
        assert lookup_stages[0].get("collectionScans", 0) == 0

    def test_attendance_list_uses_index(self, client):
        r = client.get("/admin/explain/attendance_list", params={
            "emp_code": "EMP0001",
            "date_from": "2026-07-01",
            "date_to": "2026-07-31",
        })
        assert r.status_code == 200
        data = r.json()
        assert "explain" in data
        exp = data["explain"]
        assert _find_explain_stages(exp, "COLLSCAN") == []
        ixscans = _find_explain_stages(exp, "IXSCAN")
        assert len(ixscans) >= 1
        assert ixscans[0].get("indexName") in ("date_-1_emp_code_1", "emp_code_1_date_1")

    def test_explain_attendance_list_validates_dates_and_status(self, client):
        # Invalid calendar date
        r = client.get("/admin/explain/attendance_list", params={"date_from": "2026-02-31"})
        assert r.status_code == 422
        # date_from > date_to
        r = client.get("/admin/explain/attendance_list", params={"date_from": "2026-07-15", "date_to": "2026-07-10"})
        assert r.status_code == 422
        # Invalid status
        r = client.get("/admin/explain/attendance_list", params={"status": "INVALID"})
        assert r.status_code == 422

    def test_explain_employee_monthly_pipeline_fidelity(self, client):
        r = client.get("/admin/explain/employee_monthly", params={
            "emp_code": "EMP0001",
            "month": "2026-07",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "employee_monthly"
        assert "explain" in data
        exp = data["explain"]
        assert _find_explain_stages(exp, "COLLSCAN") == []
        ixscans = _find_explain_stages(exp, "IXSCAN")
        assert len(ixscans) >= 1
        assert ixscans[0].get("indexName") == "emp_code_1_date_1"

    def test_explain_late_leaderboard_pipeline_fidelity(self, client):
        r = client.get("/admin/explain/late_leaderboard", params={
            "month": "2026-07",
            "department": "Engineering",
            "limit": 5,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "late_leaderboard"
        assert "explain" in data
        exp = data["explain"]
        assert _find_explain_stages(exp, "COLLSCAN") == []
        assert len(_find_explain_stages(exp, "IXSCAN")) >= 1
        lookup_stages = [s for s in exp.get("stages", []) if any("lookup" in k for k in s.keys())]
        assert len(lookup_stages) >= 1
        assert "indexesUsed" in lookup_stages[0]
        assert lookup_stages[0].get("collectionScans", 0) == 0

    def test_explain_department_trend_pipeline_fidelity(self, client):
        client.post("/employees", json={
            "emp_code": "EMP0001",
            "name": "Eng Test",
            "email": "engtest@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        r = client.get("/admin/explain/department_trend", params={
            "department": "Engineering",
            "from": "2026-07-01",
            "to": "2026-07-07",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["endpoint"] == "department_trend"
        assert data["collection"] == "employees"
        assert "explain" in data
        exp = data["explain"]
        # Department trend executes aggregate on employees collection;
        # Root plan must contain literal IXSCAN on department_1_emp_code_1 with 0 COLLSCAN,
        # and lookup into attendance_logs must use date_-1_emp_code_1 index.
        assert _find_explain_stages(exp, "COLLSCAN") == []
        ixscans = _find_explain_stages(exp, "IXSCAN")
        assert len(ixscans) >= 1
        assert ixscans[0].get("indexName") == "department_1_emp_code_1"
        stages = exp.get("stages", [])
        lookup_stages = [s for s in stages if any("lookup" in k for k in s.keys())]
        assert len(lookup_stages) == 1
        for s in lookup_stages:
            assert "indexesUsed" in s
            assert len(s["indexesUsed"]) >= 1
            assert s.get("collectionScans", 0) == 0

    def test_explain_department_trend_validates_dates(self, client):
        r = client.get("/admin/explain/department_trend", params={
            "department": "Engineering",
            "from": "2026-02-31",
            "to": "2026-07-07",
        })
        assert r.status_code == 422
        r = client.get("/admin/explain/department_trend", params={
            "department": "Engineering",
            "from": "2026-07-10",
            "to": "2026-07-05",
        })
        assert r.status_code == 422

    def test_explain_all_five_endpoints_comprehensive_index_verification(self, client):
        """Comprehensive verification: all 5 explain endpoints run against real MongoDB, execute index scans, and have 0 COLLSCAN."""
        client.post("/employees", json={
            "emp_code": "EMP0001",
            "name": "Eng Test",
            "email": "engtest@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        endpoints = [
            ("attendance_list", "/admin/explain/attendance_list?page=1&page_size=5"),
            ("employee_monthly", "/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07"),
            ("department_summary", "/admin/explain/department_summary?month=2026-07&department=Engineering"),
            ("late_leaderboard", "/admin/explain/late_leaderboard?month=2026-07&limit=5"),
            ("department_trend", "/admin/explain/department_trend?department=Engineering&from=2026-07-06&to=2026-07-10")
        ]
        for name, url in endpoints:
            r = client.get(url)
            assert r.status_code == 200, f"Explain failed for {name}"
            exp = r.json()["explain"]
            # Assert 0 collection scans across the board
            collscans = _find_explain_stages(exp, "COLLSCAN")
            assert collscans == [], f"Unexpected COLLSCAN in {name}: {collscans}"
            ixscans = _find_explain_stages(exp, "IXSCAN")
            assert len(ixscans) >= 1, f"Expected IXSCAN for {name}"
            if name == "department_trend":
                lookups = [s for s in exp.get("stages", []) if any("lookup" in k for k in s.keys())]
                assert len(lookups) == 1
                for lk in lookups:
                    assert len(lk.get("indexesUsed", [])) > 0
                    assert lk.get("collectionScans", 0) == 0


@skip_no_mongo
class TestConcurrency:
    def test_concurrent_punch_out_race(self, client):
        """Verify race-safe punch-out: two concurrent requests for the same session result in exactly one 200 and one 409."""
        emp_code = "EMP9981"
        client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Conc Out",
            "email": "cout@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        ts_in = int(datetime(2026, 7, 8, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        # Punch in
        p_in = client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ts_in})
        assert p_in.status_code in (201, 409)

        ts_out = int(datetime(2026, 7, 8, 13, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

        def do_punch_out():
            return client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ts_out})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(do_punch_out)
            f2 = executor.submit(do_punch_out)
            r1 = f1.result()
            r2 = f2.result()

        statuses = sorted([r1.status_code, r2.status_code])
        assert statuses == [200, 409]

    def test_concurrent_punch_in_race(self, client):
        """Verify race-safe punch-in: two concurrent requests for the same employee and date result in exactly one 201 and one 409."""
        emp_code = "EMP9982"
        client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Conc In",
            "email": "cin@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        ts_in = int(datetime(2026, 7, 9, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

        def do_punch_in():
            return client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ts_in})

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(do_punch_in)
            f2 = executor.submit(do_punch_in)
            r1 = f1.result()
            r2 = f2.result()

        statuses = sorted([r1.status_code, r2.status_code])
        assert statuses == [201, 409]


@skip_no_mongo
class TestPhase3AuditRegressions:
    def test_punch_out_selection_and_boundaries(self, client):
        emp_code = "EMP9801"
        client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Audit Test",
            "email": "audit.test@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        ts_d1_in = int(datetime(2026, 7, 6, 3, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
        ts_d2_in = int(datetime(2026, 7, 7, 3, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)

        r_in1 = client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ts_d1_in})
        assert r_in1.status_code == 201
        r_in2 = client.post("/attendance/punch-in", json={"emp_code": emp_code, "punched_at": ts_d2_in})
        assert r_in2.status_code == 201

        # Backdated punch-out when a later record exists: closes Day 1 record
        ts_d1_out = int(datetime(2026, 7, 6, 12, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
        r_out1 = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ts_d1_out})
        assert r_out1.status_code == 200
        assert r_out1.json()["date"] == "2026-07-06"
        assert r_out1.json()["punch_out"] == ts_d1_out

        # Punching out Day 1 again returns 409
        r_out1_dup = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ts_d1_out})
        assert r_out1_dup.status_code == 409

        # Equal timestamps: punch_out == punch_in -> 422
        r_eq = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ts_d2_in})
        assert r_eq.status_code == 422

        # Timestamp before first punch-in when punch-in exists -> 422
        ts_before = ts_d1_in - 3600000
        r_bef = client.post("/attendance/punch-out", json={"emp_code": emp_code, "punched_at": ts_before})
        assert r_bef.status_code == 422

        # 24-hour boundary
        emp_code_24 = "EMP9802"
        client.post("/employees", json={
            "emp_code": emp_code_24,
            "name": "Audit Test 24",
            "email": "audit24@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        ts_in_24 = int(datetime(2026, 7, 10, 3, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)
        client.post("/attendance/punch-in", json={"emp_code": emp_code_24, "punched_at": ts_in_24})

        # 24 hours + 1 second -> 422
        r_exceed = client.post("/attendance/punch-out", json={"emp_code": emp_code_24, "punched_at": ts_in_24 + 86401000})
        assert r_exceed.status_code == 422

        # Exactly 24 hours -> 200
        r_exact = client.post("/attendance/punch-out", json={"emp_code": emp_code_24, "punched_at": ts_in_24 + 86400000})
        assert r_exact.status_code == 200

    def test_trend_range_boundaries(self, client):
        # Exactly 92 inclusive calendar days: 2026-07-01 to 2026-09-30 (31 + 31 + 30 = 92)
        r92 = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-01",
            "to": "2026-09-30",
        })
        assert r92.status_code == 200
        assert len(r92.json()["items"]) == 92

        # 93 inclusive calendar days: 2026-07-01 to 2026-10-01 -> 422
        r93 = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-01",
            "to": "2026-10-01",
        })
        assert r93.status_code == 422

        # In explain endpoint as well
        r_exp92 = client.get("/admin/explain/department_trend", params={
            "department": "Engineering",
            "from": "2026-07-01",
            "to": "2026-09-30",
        })
        assert r_exp92.status_code == 200

        r_exp93 = client.get("/admin/explain/department_trend", params={
            "department": "Engineering",
            "from": "2026-07-01",
            "to": "2026-10-01",
        })
        assert r_exp93.status_code == 422

    def test_strict_regex_trailing_newlines(self, client):
        r1 = client.post("/employees", json={
            "emp_code": "EMP9899\n",
            "name": "Newline Test",
            "email": "newline@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        assert r1.status_code == 422

        r2 = client.post("/employees", json={
            "emp_code": "EMP9899",
            "name": "Newline Test",
            "email": "newline@example.com\n",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        assert r2.status_code == 422

        r3 = client.post("/employees", json={
            "emp_code": "EMP9899",
            "name": "Newline Test",
            "email": "newline@example.com",
            "department": "Engineering",
            "shift_start": "09:00\n",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        assert r3.status_code == 422

    def test_competition_ranking_ties_and_limit_cutoff(self, client):
        from pymongo import MongoClient
        client_mongo = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db_mongo = client_mongo[os.getenv("MONGO_DB", "attendance_test_db")]
        dept = "TieDept"
        db_mongo.employees.insert_many([
            {"emp_code": "EMPT001", "name": "Tie 1", "email": "t1@ex.com", "department": dept, "shift_start": "09:00", "shift_end": "18:00", "joined_on": "2026-07-01", "created_at": datetime.now(timezone.utc)},
            {"emp_code": "EMPT002", "name": "Tie 2", "email": "t2@ex.com", "department": dept, "shift_start": "09:00", "shift_end": "18:00", "joined_on": "2026-07-01", "created_at": datetime.now(timezone.utc)},
            {"emp_code": "EMPT003", "name": "Tie 3", "email": "t3@ex.com", "department": dept, "shift_start": "09:00", "shift_end": "18:00", "joined_on": "2026-07-01", "created_at": datetime.now(timezone.utc)},
        ])
        db_mongo.attendance_logs.insert_many([
            {"emp_code": "EMPT001", "date": "2026-07-06", "status": "PRESENT", "late_minutes": 30},
            {"emp_code": "EMPT002", "date": "2026-07-06", "status": "PRESENT", "late_minutes": 20},
            {"emp_code": "EMPT003", "date": "2026-07-06", "status": "PRESENT", "late_minutes": 20},
        ])
        client_mongo.close()

        r = client.get("/analytics/leaderboard/late", params={"month": "2026-07", "department": dept, "limit": 2})
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 3
        assert items[0]["emp_code"] == "EMPT001"
        assert items[0]["rank"] == 1
        assert items[1]["rank"] == 2
        assert items[2]["rank"] == 2
        assert items[1]["emp_code"] < items[2]["emp_code"]

    def test_trend_moving_average_rounding_boundary(self, client):
        """Verify moving_avg_7d calculates the mean of defined non-null 4-decimal attendance_rate values."""
        from pymongo import MongoClient
        client_mongo = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db_mongo = client_mongo[os.getenv("MONGO_DB", "attendance_test_db")]
        dept = "RoundingDept"
        # Insert 7 employees joining on 2026-07-01
        db_mongo.employees.insert_many([
            {
                "emp_code": f"EMP991{i}",
                "name": f"Round {i}",
                "email": f"r{i}@ex.com",
                "department": dept,
                "shift_start": "09:00",
                "shift_end": "18:00",
                "joined_on": "2026-07-01",
                "created_at": datetime.now(timezone.utc),
            }
            for i in range(1, 8)
        ])
        # Day 1 (Monday 2026-07-06): 3 present out of 7 -> 3/7 = 0.428571... -> rounded 4-dec is 0.4286
        # Day 2 (Tuesday 2026-07-07): 2 present out of 7 -> 2/7 = 0.285714... -> rounded 4-dec is 0.2857
        # Mean of rounded rates = (0.4286 + 0.2857)/2 = 0.35715 -> rounded 4-dec half-up is 0.3572
        # (Note: mean of unrounded fractions would be 5/14 = 0.357142... -> 0.3571)
        db_mongo.attendance_logs.insert_many([
            {"emp_code": "EMP9911", "date": "2026-07-06", "status": "PRESENT", "punch_in": 1, "punch_out": 2},
            {"emp_code": "EMP9912", "date": "2026-07-06", "status": "PRESENT", "punch_in": 1, "punch_out": 2},
            {"emp_code": "EMP9913", "date": "2026-07-06", "status": "PRESENT", "punch_in": 1, "punch_out": 2},
            {"emp_code": "EMP9911", "date": "2026-07-07", "status": "PRESENT", "punch_in": 1, "punch_out": 2},
            {"emp_code": "EMP9912", "date": "2026-07-07", "status": "PRESENT", "punch_in": 1, "punch_out": 2},
        ])
        client_mongo.close()

        r = client.get(f"/analytics/departments/{dept}/trend", params={
            "from": "2026-07-05",  # Sunday (non-working day)
            "to": "2026-07-07",
        })
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 3

        # Sunday: non-working day -> null rate and null moving average
        assert items[0]["date"] == "2026-07-05"
        assert items[0]["is_working_day"] is False
        assert items[0]["attendance_rate"] is None
        assert items[0]["moving_avg_7d"] is None

        # Monday: 3/7 -> 0.4286
        assert items[1]["date"] == "2026-07-06"
        assert items[1]["is_working_day"] is True
        assert items[1]["headcount"] == 7
        assert items[1]["present_count"] == 3
        assert items[1]["attendance_rate"] == 0.4286
        assert items[1]["moving_avg_7d"] == 0.4286

        # Tuesday: 2/7 -> 0.2857; moving average = (0.4286 + 0.2857)/2 = 0.35715 -> 0.3572
        assert items[2]["date"] == "2026-07-07"
        assert items[2]["is_working_day"] is True
        assert items[2]["headcount"] == 7
        assert items[2]["present_count"] == 2
        assert items[2]["attendance_rate"] == 0.2857
        assert items[2]["moving_avg_7d"] == 0.3572

    def test_monthly_analytics_mid_month_joiner_and_no_logs(self, client):
        """Verify numerical accuracy for mid-month joiners and employees with 0 logs."""
        # Employee with 0 logs
        emp_nologs = "EMP9983"
        client.post("/employees", json={
            "emp_code": emp_nologs,
            "name": "No Logs",
            "email": "nologs@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-01",
        })
        r_empty = client.get(f"/analytics/employees/{emp_nologs}/monthly", params={"month": "2026-07"})
        assert r_empty.status_code == 200
        data_empty = r_empty.json()
        assert data_empty["emp_code"] == emp_nologs
        assert data_empty["working_days"] == 23
        assert data_empty["present_days"] == 0
        assert data_empty["leave_days"] == 0
        assert data_empty["late_count"] == 0
        assert data_empty["total_late_minutes"] == 0
        assert data_empty["total_overtime_minutes"] == 0
        assert data_empty["attendance_pct"] == 0.0

        # Mid-month joiner: joined 2026-07-15
        emp_mid = "EMP9984"
        client.post("/employees", json={
            "emp_code": emp_mid,
            "name": "Mid Joiner",
            "email": "mid@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "18:00",
            "joined_on": "2026-07-15",
        })
        # Add 1 full day log (2026-07-16) and 1 half-day log with late and overtime (2026-07-17)
        from pymongo import MongoClient
        client_mongo = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db_mongo = client_mongo[os.getenv("MONGO_DB", "attendance_test_db")]
        db_mongo.attendance_logs.insert_many([
            {
                "emp_code": emp_mid,
                "date": "2026-07-16",
                "status": "PRESENT",
                "half_day": False,
                "late_minutes": 0,
                "overtime_minutes": 0,
            },
            {
                "emp_code": emp_mid,
                "date": "2026-07-17",
                "status": "PRESENT",
                "half_day": True,
                "late_minutes": 25,
                "overtime_minutes": 45,
            },
        ])
        client_mongo.close()

        r_mid = client.get(f"/analytics/employees/{emp_mid}/monthly", params={"month": "2026-07"})
        assert r_mid.status_code == 200
        data_mid = r_mid.json()
        assert data_mid["emp_code"] == emp_mid
        assert data_mid["working_days"] == 13
        assert data_mid["present_days"] == 1.5
        assert data_mid["leave_days"] == 0
        assert data_mid["late_count"] == 1
        assert data_mid["total_late_minutes"] == 25
        assert data_mid["total_overtime_minutes"] == 45
        assert data_mid["attendance_pct"] == 11.54

    def test_validation_error_responses_consistency(self, client):
        """Verify both Pydantic schema validation and handler-level validation return 422 with a detail key."""
        # 1. Pydantic schema validation error (missing required fields)
        r_schema = client.post("/employees", json={"emp_code": "INV"})
        assert r_schema.status_code == 422
        assert "detail" in r_schema.json()

        # 2. Handler-level validation error (logical date inversion)
        r_handler = client.get("/analytics/departments/Engineering/trend", params={
            "from": "2026-07-10",
            "to": "2026-07-05",
        })
        assert r_handler.status_code == 422
        assert "detail" in r_handler.json()
        assert r_handler.json()["detail"] == "'to' must not be before 'from'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
