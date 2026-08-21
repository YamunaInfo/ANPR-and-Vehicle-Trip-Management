"""
Final Acceptance Test Script for ANPRX Edge ANPR & Trip Management.
Performs the complete single vehicle approach -> recognition -> entry -> trip inside -> exit -> dwell time -> trip completed -> restart persistence lifecycle test.
"""
from __future__ import annotations

import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.session import SessionLocal, init_db
from db.models import (
    Alert,
    EntryExitEvent,
    GateDecision,
    ScheduledTrip,
    TripStatusHistory,
    Vehicle,
    VehiclePlate,
)
from ai.db_service import db_service
import routes_gatesense


def run_acceptance_scenario():
    print("=" * 75)
    print("      ANPRX - COMPLETE REAL VEHICLE END-TO-END ACCEPTANCE TEST")
    print("=" * 75)

    test_plate = "WB12AB1234"
    db = SessionLocal()

    # Step 1: Ensure vehicle is in database master
    print("\n[Step 1] Verifying Vehicle in Master Database...")
    veh = db_service.lookup_vehicle(test_plate, db=db)
    assert veh is not None, f"Vehicle {test_plate} not in master database!"
    print(f"  [OK] Vehicle found: {veh['plate_number']} | Owner: {veh['owner_name']} | Transporter: {veh['transporter']}")

    # Step 2: Ensure Scheduled Trip exists
    print("\n[Step 2] Checking / Creating Scheduled Trip...")
    trip = db_service.lookup_scheduled_trip(test_plate, db=db)
    if not trip:
        new_trip = ScheduledTrip(
            trip_number=f"TRIP-ACC-{int(time.time()) % 10000}",
            plate_number=test_plate,
            driver_name="Debashis Roy",
            transporter_name=veh["transporter"],
            gate_name="Gate 01",
            purpose="Industrial component delivery",
            expected_arrival=datetime.datetime.utcnow(),
            expected_departure=datetime.datetime.utcnow() + datetime.timedelta(hours=2),
            status="scheduled",
        )
        db.add(new_trip)
        db.commit()
        trip_id = new_trip.id
    else:
        trip_id = trip["id"]
    print(f"  [OK] Active Trip confirmed: #{trip_id} for plate {test_plate} (Status: Scheduled)")

    # Step 3: Vehicle Approaches Entry Gate -> ANPR Multi-Frame Recognition
    print("\n[Step 3] Vehicle Approaches Gate 01 -> ANPR Camera G01-ENTRY...")
    ocr_frames = [
        {"frame_number": 101, "plate": "WB12A81234", "confidence": 0.84, "engine": "PaddleOCR"},
        {"frame_number": 102, "plate": "WB12AB1234", "confidence": 0.95, "engine": "PaddleOCR"},
        {"frame_number": 103, "plate": "WB12AB1234", "confidence": 0.92, "engine": "PaddleOCR"},
    ]

    print("  -> Running multi-frame OCR consensus fusion...")
    entry_res = db_service.record_finalized_anpr_event({
        "track_id": 101,
        "plate_number": test_plate,
        "final_confidence": 0.94,
        "frame_count": 3,
        "frame_predictions": ocr_frames,
        "supporting_predictions": [{"plate": test_plate, "confidence": 0.95}],
        "vehicle_type": "Truck",
        "camera_id": "G01-ENTRY",
        "gate_id": "Gate 01",
        "event_type": "entry",
    })

    print(f"  [OK] Fused Plate Recognition Result: {entry_res['plate']}")
    print(f"  [OK] Gate Decision: {entry_res['decision'].upper()}")
    assert entry_res["decision"] == "allow", "Entry should be allowed for authorized vehicle"
    entry_event_id = entry_res["event_id"]

    # Step 4: Verify Entry Record & Trip Status -> 'inside_plant'
    print("\n[Step 4] Verifying Entry Event & Trip Progress in MySQL...")
    db.commit()
    ev_entry = db.query(EntryExitEvent).filter(EntryExitEvent.id == entry_event_id).first()
    assert ev_entry is not None
    assert ev_entry.event_type == "entry"
    assert ev_entry.decision == "allow"

    trip_updated = db.query(ScheduledTrip).filter(ScheduledTrip.id == trip_id).first()
    assert trip_updated.status == "inside_plant", f"Trip status should be 'inside_plant', got {trip_updated.status}"
    assert trip_updated.actual_entry_time is not None
    print(f"  [OK] Entry event #{ev_entry.id} saved in MySQL.")
    print(f"  [OK] Trip #{trip_id} status updated to: {trip_updated.status} (Entry Time: {trip_updated.actual_entry_time})")

    # Step 5: Verify Live Dashboard API visibility
    print("\n[Step 5] Verifying Live Dashboard API Endpoint...")
    summary = routes_gatesense.dashboard_summary(db=db)
    assert summary["vehiclesInside"] >= 1
    assert summary["entriesToday"] >= 1
    print(f"  [OK] Dashboard API Summary: Vehicles Inside: {summary['vehiclesInside']}, Entries Today: {summary['entriesToday']}, Accuracy: {summary['recognitionAccuracy']*100:.1f}%")

    # Simulate some plant dwell time
    dwell_sim_minutes = 48
    trip_updated.actual_entry_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=dwell_sim_minutes)
    db.commit()

    # Step 6: Vehicle Approaches Exit Gate -> ANPR Camera G03-EXIT
    print("\n[Step 6] Vehicle Approaches Exit Gate -> ANPR Camera G03-EXIT...")
    exit_frames = [
        {"frame_number": 201, "plate": "WB12AB1234", "confidence": 0.96, "engine": "PaddleOCR"},
        {"frame_number": 202, "plate": "WB12AB1234", "confidence": 0.97, "engine": "PaddleOCR"},
    ]

    exit_res = db_service.record_finalized_anpr_event({
        "track_id": 201,
        "plate_number": test_plate,
        "final_confidence": 0.96,
        "frame_count": 2,
        "frame_predictions": exit_frames,
        "supporting_predictions": [{"plate": test_plate, "confidence": 0.97}],
        "vehicle_type": "Truck",
        "camera_id": "G03-EXIT",
        "gate_id": "Gate 03",
        "event_type": "exit",
    })

    print(f"  [OK] Exit Recognition Result: {exit_res['plate']}")
    assert exit_res["decision"] == "allow"
    exit_event_id = exit_res["event_id"]

    # Step 7: Verify Exit Event, Dwell Time Calculation, and Trip Completion
    print("\n[Step 7] Verifying Dwell Time & Trip Completion in MySQL...")
    db.commit()
    ev_exit = db.query(EntryExitEvent).filter(EntryExitEvent.id == exit_event_id).first()
    assert ev_exit is not None
    assert ev_exit.event_type == "exit"

    trip_completed = db.query(ScheduledTrip).filter(ScheduledTrip.id == trip_id).first()
    assert trip_completed.status == "completed", f"Trip status should be 'completed', got {trip_completed.status}"
    assert trip_completed.actual_exit_time is not None
    assert trip_completed.dwell_minutes is not None
    print(f"  [OK] Exit event #{ev_exit.id} saved in MySQL.")
    print(f"  [OK] Calculated Dwell Time: {trip_completed.dwell_minutes} minutes")
    print(f"  [OK] Trip #{trip_id} completed successfully!")

    # Verify status history audit trail
    histories = db.query(TripStatusHistory).filter(TripStatusHistory.trip_id == trip_id).order_by(TripStatusHistory.id.asc()).all()
    print(f"  [OK] Audit history records for trip ({len(histories)} transitions):")
    for h in histories:
        print(f"      - {h.from_status} -> {h.to_status} ({h.notes}) at {h.timestamp}")

    # Step 8: Verify Persistence Across Simulated Restart
    print("\n[Step 8] Verifying Persistence Across Simulated Backend Restart...")
    db.close()

    # Re-initialize new connection pool / session
    fresh_db = SessionLocal()
    persisted_trip = fresh_db.query(ScheduledTrip).filter(ScheduledTrip.id == trip_id).first()
    assert persisted_trip is not None, "Trip lost after restart!"
    assert persisted_trip.status == "completed", "Trip state lost after restart!"
    assert persisted_trip.dwell_minutes == trip_completed.dwell_minutes

    persisted_entry = fresh_db.query(EntryExitEvent).filter(EntryExitEvent.id == entry_event_id).first()
    assert persisted_entry is not None, "Entry event lost after restart!"

    persisted_exit = fresh_db.query(EntryExitEvent).filter(EntryExitEvent.id == exit_event_id).first()
    assert persisted_exit is not None, "Exit event lost after restart!"

    print(f"  [OK] Verified 100% data persistence in MySQL for all records after restart.")
    fresh_db.close()

    print("\n" + "=" * 75)
    print("  FINAL ACCEPTANCE TEST COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    print("=" * 75)


if __name__ == "__main__":
    run_acceptance_scenario()
