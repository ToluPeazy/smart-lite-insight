"""Tests for the Phase 1 base-load estimator (src/baseload.py)."""

import json
import sys
from datetime import datetime, timedelta

import jsonschema
import pytest

from src.baseload import compute_base_load
from src.ingest import init_db

SITE_ID = "home-01"
SCHEMA_PATH = "docs/schemas/baseload_v1.json"


def _insert_reading(conn, ts, site_id, global_kw, sub1_wh, sub2_wh, sub3_wh):
    conn.execute(
        """
        INSERT INTO readings (
            timestamp, site_id, device_id,
            global_active_power_kw, global_reactive_power_kw,
            voltage_v, global_intensity_a,
            sub_metering_1_wh, sub_metering_2_wh, sub_metering_3_wh
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts.isoformat(),
            site_id,
            "meter-main",
            global_kw,
            0.0,
            230.0,
            1.0,
            sub1_wh,
            sub2_wh,
            sub3_wh,
        ),
    )


def _build_two_day_fixture(db_path, site_id=SITE_ID):
    """Two full days (48 half-hourly readings each) of constant per-day power.

    Day 1 every slot = 1.0 kW, day 2 every slot = 2.0 kW, no sub-metering
    load, so the submeter_subtraction mean per slot is (1.0+2.0)/2 = 1.5 kW
    and the quantile-p25 per slot is 1.25 kW (linear interpolation between
    1.0 and 2.0) -- both hand-checkable.
    """
    conn = init_db(db_path)
    start_day = datetime(2007, 1, 1)
    for day_offset, power_kw in enumerate([1.0, 2.0]):
        day = start_day + timedelta(days=day_offset)
        for slot in range(48):
            ts = day + timedelta(minutes=slot * 30)
            _insert_reading(conn, ts, site_id, power_kw, 0.0, 0.0, 0.0)
    conn.commit()
    conn.close()


def _build_subtraction_fixture(db_path, site_id=SITE_ID):
    """Single day, 48 slots, with sub-metering load on two specific slots.

    Slot 5 (02:30): global=5.0kW, sub1=50Wh, sub2=30Wh
        -> flexible_kw = (50+30) * 0.06 = 4.8kW -> base = 0.2kW
    Slot 6 (03:00): global=1.0kW, sub1=200Wh, sub2=200Wh
        -> flexible_kw = (200+200) * 0.06 = 24.0kW -> base clips to 0.0kW
    All other slots: global=1.0kW, no sub-metering load -> base = 1.0kW
    """
    conn = init_db(db_path)
    day = datetime(2007, 1, 1)
    for slot in range(48):
        ts = day + timedelta(minutes=slot * 30)
        if slot == 5:
            _insert_reading(conn, ts, site_id, 5.0, 50.0, 30.0, 0.0)
        elif slot == 6:
            _insert_reading(conn, ts, site_id, 1.0, 200.0, 200.0, 0.0)
        else:
            _insert_reading(conn, ts, site_id, 1.0, 0.0, 0.0, 0.0)
    conn.commit()
    conn.close()


# ── submeter_subtraction method ──


def test_submeter_subtraction_computes_expected_values(tmp_db):
    _build_subtraction_fixture(tmp_db)

    profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, method="submeter_subtraction"
    )

    slots = {s["slot"]: s for s in profile["slots"]}
    assert slots[5]["base_load_kw"] == pytest.approx(0.2)
    assert slots[0]["base_load_kw"] == pytest.approx(1.0)


def test_submeter_subtraction_clips_negative_to_zero(tmp_db):
    _build_subtraction_fixture(tmp_db)

    profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, method="submeter_subtraction"
    )

    slots = {s["slot"]: s for s in profile["slots"]}
    assert slots[6]["base_load_kw"] == 0.0


def test_submeter_subtraction_averages_across_days(tmp_db):
    _build_two_day_fixture(tmp_db)

    profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, method="submeter_subtraction"
    )

    for s in profile["slots"]:
        assert s["base_load_kw"] == pytest.approx(1.5)


# ── quantile method ──


def test_quantile_method_computes_expected_value(tmp_db):
    _build_two_day_fixture(tmp_db)

    profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, method="quantile", quantile=0.25
    )

    for s in profile["slots"]:
        assert s["base_load_kw"] == pytest.approx(1.25)


def test_quantile_default_is_p25(tmp_db):
    _build_two_day_fixture(tmp_db)

    default_profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, method="quantile"
    )
    explicit_profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, method="quantile", quantile=0.25
    )

    assert default_profile["slots"] == explicit_profile["slots"]


# ── kWh conversion, shape, determinism ──


def test_kwh_is_half_of_kw(tmp_db):
    _build_two_day_fixture(tmp_db)

    profile = compute_base_load(db_path=tmp_db, site_id=SITE_ID)

    for s in profile["slots"]:
        assert s["base_load_kwh"] == pytest.approx(s["base_load_kw"] * 0.5)


def test_profile_has_48_sequential_slots(tmp_db):
    _build_two_day_fixture(tmp_db)

    profile = compute_base_load(db_path=tmp_db, site_id=SITE_ID)

    assert [s["slot"] for s in profile["slots"]] == list(range(48))
    assert profile["slots"][0]["start"] == "00:00"
    assert profile["slots"][1]["start"] == "00:30"
    assert profile["slots"][47]["start"] == "23:30"


def test_profile_metadata(tmp_db):
    _build_two_day_fixture(tmp_db)

    profile = compute_base_load(db_path=tmp_db, site_id=SITE_ID, method="quantile")

    assert profile["schema_version"] == "baseload-1.0"
    assert profile["site_id"] == SITE_ID
    assert profile["method"] == "quantile"
    assert profile["source_range"] == {"start": "2007-01-01", "end": "2007-01-02"}


def test_deterministic(tmp_db):
    _build_two_day_fixture(tmp_db)

    first = compute_base_load(db_path=tmp_db, site_id=SITE_ID)
    second = compute_base_load(db_path=tmp_db, site_id=SITE_ID)

    assert first == second


def test_invalid_method_raises(tmp_db):
    _build_two_day_fixture(tmp_db)

    with pytest.raises(ValueError, match="Unknown method"):
        compute_base_load(db_path=tmp_db, site_id=SITE_ID, method="bogus")


def test_missing_slot_raises(tmp_db):
    # Only insert slot 0 -- the other 47 slots are missing entirely.
    conn = init_db(tmp_db)
    _insert_reading(conn, datetime(2007, 1, 1, 0, 0), SITE_ID, 1.0, 0.0, 0.0, 0.0)
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="No data for slot"):
        compute_base_load(db_path=tmp_db, site_id=SITE_ID)


def test_no_readings_for_site_raises(tmp_db):
    init_db(tmp_db).close()

    with pytest.raises(ValueError, match="No readings found"):
        compute_base_load(db_path=tmp_db, site_id="nonexistent-site")


def test_date_range_filters_readings(tmp_db):
    _build_two_day_fixture(tmp_db)

    profile = compute_base_load(
        db_path=tmp_db, site_id=SITE_ID, start="2007-01-02", end="2007-01-02T23:59:59"
    )

    # Only day 2 (2.0 kW every slot) is in range.
    for s in profile["slots"]:
        assert s["base_load_kw"] == pytest.approx(2.0)
    assert profile["source_range"] == {
        "start": "2007-01-02",
        "end": "2007-01-02T23:59:59",
    }


# ── schema validation ──


def test_output_validates_against_published_schema(tmp_db, project_root):
    _build_two_day_fixture(tmp_db)
    profile = compute_base_load(db_path=tmp_db, site_id=SITE_ID)

    with open(project_root / SCHEMA_PATH) as f:
        schema = json.load(f)

    jsonschema.validate(instance=profile, schema=schema)


# ── CLI ──


def test_cli_prints_json_to_stdout(tmp_db, monkeypatch, capsys):
    _build_two_day_fixture(tmp_db)
    monkeypatch.setattr(
        sys, "argv", ["baseload", "--db-path", tmp_db, "--site-id", SITE_ID]
    )

    from src.baseload import main

    main()

    out = capsys.readouterr().out
    profile = json.loads(out)
    assert profile["schema_version"] == "baseload-1.0"
    assert len(profile["slots"]) == 48


def test_cli_writes_to_output_file(tmp_db, monkeypatch, tmp_path):
    _build_two_day_fixture(tmp_db)
    out_path = tmp_path / "profile.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "baseload",
            "--db-path",
            tmp_db,
            "--site-id",
            SITE_ID,
            "--method",
            "quantile",
            "--output",
            str(out_path),
        ],
    )

    from src.baseload import main

    main()

    profile = json.loads(out_path.read_text())
    assert profile["method"] == "quantile"
    assert len(profile["slots"]) == 48
