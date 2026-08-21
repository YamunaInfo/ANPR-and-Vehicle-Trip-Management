"""
Database Storage Inspector & Verification Script for ANPRX / GateSense

This script verifies:
1. MySQL Database connection and table integrity.
2. Row counts across all 24 tables.
3. Recent live entry/exit events, OCR plate predictions, and detections stored.
4. Active trips, whitelist/blacklist rules, and system alerts.
5. A live Write-and-Read test verifying that data insertion is functioning properly.
"""
from __future__ import annotations

import datetime
import os
import sys

# Configure UTF-8 safe stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import func, inspect, text
from db.config import get_database_url, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PORT
from db.session import SessionLocal, engine, verify_connectivity
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


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_database():
    print_banner("ANPRX DATABASE STORAGE VERIFICATION")
    print(f"Target Database: {MYSQL_DATABASE} on {MYSQL_HOST}:{MYSQL_PORT}")
    
    # 1. Connectivity Check
    print("\n[1] Checking Database Connectivity...")
    if not verify_connectivity(max_retries=3, retry_delay=0.5):
        print("[-] FAILED: Cannot connect to MySQL server. Please ensure MySQL is running.")
        return False
    print("[+] Database is ONLINE and reachable.")

    db = SessionLocal()
    try:
        # 2. Table Row Counts
        print_banner("TABLE ROW COUNTS (24 TABLES)")
        
        tables = [
            ("Roles", Role),
            ("Users", User),
            ("Transporters", Transporter),
            ("Vehicles", Vehicle),
            ("Vehicle Plates", VehiclePlate),
            ("Drivers", Driver),
            ("Gates", Gate),
            ("Cameras", Camera),
            ("Model Versions", ModelVersion),
            ("Scheduled Trips", ScheduledTrip),
            ("Trip Status History", TripStatusHistory),
            ("Vehicle Detections", VehicleDetection),
            ("Plate Predictions", PlatePrediction),
            ("Plate Prediction Frames", PlatePredictionFrame),
            ("Entry/Exit Events", EntryExitEvent),
            ("Gate Decisions", GateDecision),
            ("Whitelist Entries", WhitelistEntry),
            ("Watchlist (Blacklist)", WatchlistEntry),
            ("Alerts", Alert),
            ("Alert Deliveries", AlertDelivery),
            ("Audit Logs", AuditLog),
            ("Manual Corrections", ManualCorrection),
            ("Camera Health Logs", CameraHealth),
            ("Daily Gate Summary", DailyGateSummary),
        ]

        total_records = 0
        print(f"{'Table Name':<28} | {'Row Count':<10}")
        print("-" * 42)
        for name, model in tables:
            try:
                count = db.query(model).count()
                total_records += count
                print(f"{name:<28} | {count:<10}")
            except Exception as e:
                print(f"{name:<28} | Error: {e}")

        print("-" * 42)
        print(f"{'TOTAL DATABASE RECORDS':<28} | {total_records:<10}")

        # 3. Recent Entry/Exit Events
        print_banner("LATEST 5 ENTRY / EXIT EVENTS STORED")
        events = db.query(EntryExitEvent).order_by(EntryExitEvent.created_at.desc()).limit(5).all()
        if not events:
            print("No entry/exit events stored yet.")
        else:
            print(f"{'ID':<6} | {'Plate Number':<14} | {'Decision':<14} | {'Type':<8} | {'Confidence':<10} | {'Timestamp'}")
            print("-" * 75)
            for ev in events:
                ts = ev.created_at.strftime("%Y-%m-%d %H:%M:%S") if ev.created_at else "N/A"
                conf = f"{ev.confidence:.1%}" if ev.confidence is not None else "N/A"
                ev_type = ev.event_type or "entry"
                print(f"{ev.id:<6} | {ev.plate_number:<14} | {ev.decision:<14} | {ev_type:<8} | {conf:<10} | {ts}")

        # 4. Latest OCR Plate Predictions
        print_banner("LATEST 5 OCR PLATE PREDICTIONS STORED")
        predictions = db.query(PlatePrediction).order_by(PlatePrediction.created_at.desc()).limit(5).all()
        if not predictions:
            print("No plate predictions stored yet.")
        else:
            print(f"{'ID':<6} | {'Fused Plate':<14} | {'Clean Plate':<14} | {'Confidence':<10} | {'Frames':<7} | {'Status':<12}")
            print("-" * 75)
            for p in predictions:
                conf = f"{p.confidence:.1%}" if p.confidence is not None else "N/A"
                raw = p.raw_plate_text or "-"
                clean = p.clean_plate_text or "-"
                fused = p.fused_plate_text or clean
                status = p.status or "finalized"
                frames = p.frame_count or 1
                print(f"{p.id:<6} | {fused:<14} | {clean:<14} | {conf:<10} | {frames:<7} | {status:<12}")

        # 5. Live Storage Insert & Retrieve Test
        print_banner("LIVE DATA STORAGE TEST (INSERT & RETRIEVE)")
        test_plate_str = f"CHECK{datetime.datetime.now().strftime('%H%M%S')}"
        print(f"Creating test vehicle with plate: {test_plate_str}...")

        # Create test vehicle and plate
        test_v = Vehicle(
            vehicle_type="TestRunner",
            owner_name="Automated Storage Test",
            is_authorized=True,
            status="Available",
        )
        db.add(test_v)
        db.flush()

        test_vp = VehiclePlate(
            vehicle_id=test_v.id,
            plate_number=test_plate_str,
            is_primary=True,
        )
        db.add(test_vp)
        
        # Add test audit log
        test_audit = AuditLog(
            action="storage_verification_test",
            entity_type="vehicle",
            entity_id=test_v.id,
            details=f"Live storage verification test for {test_plate_str}",
        )
        db.add(test_audit)
        db.commit()

        # Retrieve and verify
        retrieved_plate = db.query(VehiclePlate).filter(VehiclePlate.plate_number == test_plate_str).first()
        retrieved_vehicle = db.query(Vehicle).filter(Vehicle.id == test_v.id).first()
        retrieved_audit = db.query(AuditLog).filter(AuditLog.entity_id == test_v.id).first()

        if retrieved_plate and retrieved_vehicle and retrieved_audit:
            print(f"[OK] INSERT SUCCESS: Vehicle (ID={retrieved_vehicle.id}) & Plate '{retrieved_plate.plate_number}' stored.")
            print(f"[OK] QUERY SUCCESS: Retrieved record successfully from MySQL.")
            print(f"[OK] AUDIT LOG SUCCESS: Stored test audit log (ID={retrieved_audit.id}).")
            
            # Clean up test entry
            db.delete(test_audit)
            db.delete(test_vp)
            db.delete(test_v)
            db.commit()
            print("[OK] CLEANUP SUCCESS: Test records removed.")
            print("\n>>> DATABASE STORAGE IS FULLY OPERATIONAL AND PERSISTING DATA! <<<")
            return True
        else:
            print("[-] Storage test verification failed to retrieve the inserted record.")
            return False

    except Exception as exc:
        print(f"\n[-] Error during database check: {exc}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = check_database()
    sys.exit(0 if success else 1)
