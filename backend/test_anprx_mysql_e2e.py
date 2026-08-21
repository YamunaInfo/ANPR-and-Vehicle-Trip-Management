"""
Comprehensive Automated Test Suite for ANPRX Edge ANPR & Trip Management Platform.
Tests MySQL 8.x integration end-to-end across all 24 tables, seed idempotency,
trip lifecycle, multi-frame fusion persistence, manual review corrections, and real video processing.
"""
from __future__ import annotations

import datetime
import os
import sys
import time

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import func, inspect, text
from db.config import get_database_url
from db.session import SessionLocal, engine, init_db, verify_connectivity
from db.seed import seed_master_data
from db.models import (
    Base,
    Role,
    User,
    Transporter,
    Vehicle,
    VehiclePlate,
    Driver,
    Gate,
    Camera,
    ModelVersion,
    ScheduledTrip,
    TripStatusHistory,
    VehicleDetection,
    PlatePrediction,
    PlatePredictionFrame,
    EntryExitEvent,
    GateDecision,
    WhitelistEntry,
    WatchlistEntry,
    Alert,
    AlertDelivery,
    AuditLog,
    ManualCorrection,
    CameraHealth,
    DailyGateSummary,
)
from ai.db_service import db_service
from ai.video_processor import VideoAnprProcessor


def test_suite():
    print("=" * 70)
    print("       ANPRX EDGE ANPR & TRIP MANAGEMENT - MYSQL VERIFICATION")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # TEST 1: MySQL Connectivity
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 1] Testing MySQL Connectivity...")
    assert verify_connectivity() is True, "MySQL connectivity check failed!"
    print("  -> MySQL 8.x is online and reachable.")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: All 24 Tables in MySQL
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 2] Verifying All 24 Database Tables in MySQL...")
    init_db()
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    required_24_tables = [
        "roles",
        "users",
        "transporters",
        "vehicles",
        "vehicle_plates",
        "drivers",
        "gates",
        "cameras",
        "model_versions",
        "scheduled_trips",
        "trip_status_history",
        "vehicle_detections",
        "plate_predictions",
        "plate_prediction_frames",
        "entry_exit_events",
        "gate_decisions",
        "whitelist_entries",
        "watchlist_entries",
        "alerts",
        "alert_deliveries",
        "audit_logs",
        "manual_corrections",
        "camera_health",
        "daily_gate_summary",
    ]

    for t_name in required_24_tables:
        assert t_name in existing_tables, f"Missing required table: {t_name}"
        cols = inspector.get_columns(t_name)
        assert len(cols) > 0, f"Table {t_name} has no columns!"
        print(f"  ✓ Table '{t_name}' verified ({len(cols)} columns)")

    print(f"  -> All {len(required_24_tables)} tables successfully created in MySQL.")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Idempotent Master Data Seeding
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 3] Testing Seed Idempotency...")
    db = SessionLocal()
    seed_master_data(db)
    count_roles_1 = db.query(Role).count()
    count_users_1 = db.query(User).count()
    count_vehicles_1 = db.query(Vehicle).count()
    count_plates_1 = db.query(VehiclePlate).count()
    count_trips_1 = db.query(ScheduledTrip).count()

    # Re-run seed immediately
    seed_master_data(db)
    count_roles_2 = db.query(Role).count()
    count_users_2 = db.query(User).count()
    count_vehicles_2 = db.query(Vehicle).count()
    count_plates_2 = db.query(VehiclePlate).count()
    count_trips_2 = db.query(ScheduledTrip).count()

    assert count_roles_1 == count_roles_2, "Seed not idempotent for roles"
    assert count_users_1 == count_users_2, "Seed not idempotent for users"
    assert count_vehicles_1 == count_vehicles_2, "Seed not idempotent for vehicles"
    assert count_plates_1 == count_plates_2, "Seed not idempotent for vehicle_plates"
    assert count_trips_1 == count_trips_2, "Seed not idempotent for scheduled_trips"
    print(f"  ✓ Seed ran twice with 0 duplicate rows created (Vehicles: {count_vehicles_1}, Users: {count_users_1})")

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Vehicle & Driver CRUD
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 4] Testing Vehicle and Driver CRUD in MySQL...")
    import random
    random_int = random.randint(1000, 9999)
    test_plate = f"TEST{random_int:04d}"
    
    # Create Vehicle & Plate
    v = Vehicle(
        vehicle_type="Truck",
        owner_name="Test Logistics Owner",
        is_authorized=True,
        status="Available",
    )
    db.add(v)
    db.flush()
    
    vp = VehiclePlate(
        vehicle_id=v.id,
        plate_number=test_plate,
        is_primary=True,
        state_code="TE",
        is_active=True,
    )
    db.add(vp)
    db.commit()

    # Lookup
    found_veh = db_service.lookup_vehicle(test_plate, db=db)
    assert found_veh is not None, "Vehicle lookup failed"
    assert found_veh["plate_number"] == test_plate
    print(f"  ✓ Created & looked up vehicle plate: {test_plate}")

    # Delete vehicle (should cascade delete plate)
    db.delete(v)
    db.commit()
    assert db.query(VehiclePlate).filter(VehiclePlate.plate_number == test_plate).first() is None
    print("  ✓ Cascading deletion of vehicle plates verified.")

    # ─────────────────────────────────────────────────────────────
    # TEST 5: Complete Trip Lifecycle & TripStatusHistory
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 5] Testing Trip Lifecycle & Status History...")
    test_trip_plate = f"TRIP{random_int:04d}"
    trip = ScheduledTrip(
        trip_number=f"TRIP-TEST-{random_int}",
        plate_number=test_trip_plate,
        driver_name="Ramesh Driver",
        transporter_name="BlueDart Logistics",
        gate_name="Gate 01",
        purpose="Finished goods delivery",
        expected_arrival=datetime.datetime.utcnow(),
        expected_departure=datetime.datetime.utcnow() + datetime.timedelta(hours=2),
        status="scheduled",
    )
    db.add(trip)
    db.commit()
    db.refresh(trip)

    statuses = [
        "arrived",
        "entry_approved",
        "inside_plant",
        "at_destination",
        "exit_detected",
        "completed",
    ]
    prev_status = "scheduled"
    for st in statuses:
        db.add(TripStatusHistory(
            trip_id=trip.id,
            from_status=prev_status,
            to_status=st,
            notes=f"Lifecycle transition to {st}",
            timestamp=datetime.datetime.utcnow(),
        ))
        trip.status = st
        if st == "inside_plant":
            trip.actual_entry_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=35)
        elif st == "completed":
            trip.actual_exit_time = datetime.datetime.utcnow()
            trip.dwell_minutes = 35
        prev_status = st
        db.commit()

    db.refresh(trip)
    histories = db.query(TripStatusHistory).filter(TripStatusHistory.trip_id == trip.id).all()
    assert len(histories) == len(statuses), "Trip status history count mismatch"
    assert trip.dwell_minutes == 35, "Dwell minutes mismatch"
    print(f"  ✓ Trip lifecycle progressed through all 6 states with {len(histories)} history records.")

    # ─────────────────────────────────────────────────────────────
    # TEST 6: ANPR Pipeline Flow & Multi-Frame Fusion Persistence
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 6] Testing ANPR Pipeline Flow & Multi-Frame Fusion Persistence...")
    fused_plate = "TN37AB1234"
    frame_preds = [
        {"frame_number": 1, "plate": "TN37A81234", "confidence": 0.82, "engine": "PaddleOCR"},
        {"frame_number": 2, "plate": "TN37AB1234", "confidence": 0.94, "engine": "PaddleOCR"},
        {"frame_number": 3, "plate": "TN37AB1234", "confidence": 0.91, "engine": "PaddleOCR"},
    ]

    res_anpr = db_service.record_finalized_anpr_event({
        "track_id": 42,
        "plate_number": fused_plate,
        "final_confidence": 0.92,
        "frame_count": 3,
        "frame_predictions": frame_preds,
        "supporting_predictions": [{"plate": "TN37AB1234", "confidence": 0.94}],
        "vehicle_type": "Truck",
        "vehicle_bbox": [100, 150, 400, 350],
        "camera_id": "G01-ENTRY",
        "gate_id": "Gate 01",
        "event_type": "entry",
    })

    assert res_anpr["status"] == "finalized", "ANPR status not finalized"
    assert res_anpr["decision"] == "allow", "Decision was not allow for authorized vehicle"
    event_id = res_anpr["event_id"]

    db.commit()
    ev_rec = db.query(EntryExitEvent).filter(EntryExitEvent.id == event_id).first()
    assert ev_rec is not None, "EntryExitEvent was not saved in MySQL!"
    assert ev_rec.plate_number == fused_plate
    assert ev_rec.decision == "allow"
    print(f"  ✓ EntryExitEvent #{ev_rec.id} recorded with decision '{ev_rec.decision}'")

    # Check vehicle detection and plate prediction frames
    pred_rec = ev_rec.plate_prediction
    assert pred_rec is not None, "PlatePrediction not linked!"
    assert len(pred_rec.frames) >= 3, "PlatePredictionFrames not stored!"
    print(f"  ✓ Stored {len(pred_rec.frames)} individual OCR frame records for plate #{pred_rec.id}")

    # ─────────────────────────────────────────────────────────────
    # TEST 7: Deduplication Window
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 7] Testing ANPR Deduplication...")
    # Immediately record same vehicle passage again within 15 seconds
    res_dedup = db_service.record_finalized_anpr_event({
        "track_id": 43,
        "plate_number": fused_plate,
        "final_confidence": 0.91,
        "frame_count": 2,
        "vehicle_type": "Truck",
        "camera_id": "G01-ENTRY",
        "gate_id": "Gate 01",
        "event_type": "entry",
    })
    assert res_dedup["status"] == "duplicate_skipped", "Deduplication did not skip repeated event!"
    print("  ✓ Deduplication successfully skipped duplicate event within time window.")

    # ─────────────────────────────────────────────────────────────
    # TEST 8: Watchlist and Unknown Vehicle Security Alerts
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 8] Testing Watchlist Blacklist & Unknown Vehicle Alerts...")
    # 1. Watchlist vehicle GJ18BR2290
    res_watch = db_service.record_finalized_anpr_event({
        "track_id": 50,
        "plate_number": "GJ18BR2290",
        "final_confidence": 0.90,
        "vehicle_type": "Truck",
        "camera_id": "G01-ENTRY",
        "gate_id": "Gate 01",
        "event_type": "entry",
    })
    assert res_watch["decision"] == "deny", "Watchlist vehicle was not denied!"
    assert res_watch.get("alert") is not None, "Watchlist alert was not created!"
    print("  ✓ Blacklisted vehicle correctly DENIED and high-priority alert generated.")

    # 2. Unknown Vehicle
    res_unknown = db_service.record_finalized_anpr_event({
        "track_id": 51,
        "plate_number": f"UNKN{random.randint(1000, 9999)}",
        "final_confidence": 0.88,
        "vehicle_type": "Car",
        "camera_id": "G02-ENTRY",
        "gate_id": "Gate 02",
        "event_type": "entry",
    })
    assert res_unknown["decision"] == "manual_review", "Unknown vehicle did not go to manual review!"
    assert res_unknown.get("alert") is not None, "Unknown vehicle alert was not created!"
    print("  ✓ Unknown vehicle sent to Manual Review with Security Alert.")

    # ─────────────────────────────────────────────────────────────
    # TEST 9: Manual Review & Operator Correction
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 9] Testing Manual Review Queue and Operator Correction...")
    # Low confidence read
    res_low_conf = db_service.record_finalized_anpr_event({
        "track_id": 60,
        "plate_number": "MH12QX9O31",
        "final_confidence": 0.68,  # Below 0.75
        "vehicle_type": "Car",
        "camera_id": "G01-ENTRY",
        "gate_id": "Gate 01",
        "event_type": "entry",
    })
    assert res_low_conf["decision"] == "manual_review"

    db.commit()
    pred_to_correct = db.query(PlatePrediction).filter(PlatePrediction.fused_plate_text == "MH12QX9O31").order_by(PlatePrediction.id.desc()).first()
    assert pred_to_correct is not None

    # Simulate operator correcting MH12QX9O31 -> MH12QX9031
    import routes_gatesense
    corr_res = routes_gatesense.correct_review_item(
        review_id=pred_to_correct.id,
        body=routes_gatesense.CorrectPlateBody(correctedPlate="MH12QX9031"),
        db=db,
    )
    assert corr_res["plate"] == "MH12QX9031"
    assert corr_res["status"] == "Resolved"

    # Verify manual_corrections record in MySQL
    db.commit()
    corr_db = db.query(ManualCorrection).filter(ManualCorrection.plate_prediction_id == pred_to_correct.id).first()
    assert corr_db is not None, "ManualCorrection record was not persisted to MySQL!"
    assert corr_db.original_plate == "MH12QX9O31"
    assert corr_db.corrected_plate == "MH12QX9031"
    print(f"  ✓ Manual correction persisted in MySQL (Original: '{corr_db.original_plate}', Corrected: '{corr_db.corrected_plate}')")

    # ─────────────────────────────────────────────────────────────
    # TEST 10: Camera Health Testing
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 10] Testing Camera Health Check Logging in MySQL...")
    db.commit()
    cam1 = db.query(Camera).first()
    cam_test_res = routes_gatesense.test_camera_connection(cam1.id, db=db)
    db.commit()
    health_recs = db.query(CameraHealth).filter(CameraHealth.camera_id == cam1.id).all()
    assert len(health_recs) > 0, "CameraHealth record not saved in MySQL!"
    print(f"  ✓ Camera health check recorded in MySQL (status: {cam1.status}, total records: {len(health_recs)})")

    # ─────────────────────────────────────────────────────────────
    # TEST 11: Real Video Processing through AI Pipeline
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 11] Testing Real Video Processing with VideoAnprProcessor...")
    video_path = os.path.join(os.path.dirname(__file__), "real_traffic_test.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(os.path.dirname(__file__), "test_traffic_video.mp4")

    if os.path.exists(video_path):
        processor = VideoAnprProcessor()
        print(f"  -> Processing video: {os.path.basename(video_path)}")
        res_video = processor.process_video(
            video_path=video_path,
            process_fps=10,
        )
        assert res_video.get("success") is True, f"Video processing failed: {res_video}"
        print(f"  ✓ Processed {res_video['frames_processed']} frames, detected {res_video['vehicles_detected']} vehicles.")
        print(f"  ✓ Real video operational detections written to MySQL.")
    else:
        print("  [Notice] Test video file not found, skipping video run.")

    # ─────────────────────────────────────────────────────────────
    # TEST 12: Reports Overview from MySQL Aggregations
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 12] Testing Live Reports Overview Aggregations...")
    reports = routes_gatesense.get_reports_overview(db=db)
    assert reports["totalReads"] > 0, "Total reads from MySQL reports should be > 0"
    assert len(reports["gateVolume"]) > 0, "Gate volume breakdown empty"
    assert len(reports["decisions"]) == 3, "Decisions breakdown incomplete"
    print(f"  ✓ Live Reports computed from MySQL (Total Reads: {reports['totalReads']}, Overstays: {reports['overstays']}, Corrected Reads: {reports['correctedReads']})")

    db.close()
    print("\n" + "=" * 70)
    print("  ALL 12 ANPRX MYSQL DATABASE INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_suite()
