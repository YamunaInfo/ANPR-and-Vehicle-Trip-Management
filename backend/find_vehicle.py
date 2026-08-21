"""
Vehicle Plate Lookup Tool for ANPRX / GateSense

Searches across all database tables for a specific license plate:
- Vehicle Registry & Owner records
- Entry / Exit Events & Gate Decisions
- OCR Predictions & Multi-Frame Fusion logs
- Scheduled Trips
- Security Whitelist / Watchlist entries
- Alerts
"""
from __future__ import annotations

import argparse
import os
import sys

# Configure UTF-8 safe stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import or_
from db.config import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PORT
from db.session import SessionLocal, verify_connectivity
from db.models import (
    Vehicle,
    VehiclePlate,
    EntryExitEvent,
    PlatePrediction,
    PlatePredictionFrame,
    ScheduledTrip,
    WhitelistEntry,
    WatchlistEntry,
    Alert,
    ManualCorrection,
)


def search_plate(target_plate: str):
    clean_target = "".join(ch for ch in target_plate.upper() if ch.isalnum())
    print("\n" + "=" * 70)
    print(f"  ANPRX VEHICLE SEARCH: '{target_plate}' (Clean: '{clean_target}')")
    print("=" * 70)

    if not verify_connectivity(max_retries=2, retry_delay=0.5):
        print("[-] Error: Cannot connect to MySQL database.")
        return

    db = SessionLocal()
    found_any = False

    try:
        # 1. Registered Vehicles & Plates
        print("\n[1] Registered Vehicles Table:")
        plates = db.query(VehiclePlate).join(Vehicle).filter(
            or_(
                VehiclePlate.plate_number.like(f"%{clean_target}%"),
                VehiclePlate.plate_number.like(f"%{target_plate}%"),
            )
        ).all()

        if plates:
            found_any = True
            for vp in plates:
                v = vp.vehicle
                print(f"  [FOUND] Vehicle ID: {v.id if v else 'N/A'}")
                print(f"          Plate Number:   {vp.plate_number}")
                print(f"          Vehicle Type:   {v.vehicle_type if v else 'N/A'}")
                print(f"          Owner:          {v.owner_name if v else 'N/A'}")
                print(f"          Authorized:     {v.is_authorized if v else 'N/A'}")
                print(f"          Status:         {v.status if v else 'N/A'}")
                print(f"          Registered At:  {vp.created_at}")
        else:
            print("  [-] Not found in registered vehicles table.")

        # 2. Entry / Exit Events
        print("\n[2] Entry / Exit Gate Events:")
        events = db.query(EntryExitEvent).filter(
            or_(
                EntryExitEvent.plate_number.like(f"%{clean_target}%"),
                EntryExitEvent.plate_number.like(f"%{target_plate}%"),
            )
        ).order_by(EntryExitEvent.created_at.desc()).all()

        if events:
            found_any = True
            for ev in events:
                conf = f"{ev.confidence:.1%}" if ev.confidence is not None else "N/A"
                print(f"  [FOUND] Event #{ev.id}:")
                print(f"          Plate Read:     {ev.plate_number}")
                print(f"          Event Type:     {ev.event_type} at {ev.gate_name} ({ev.camera_name})")
                print(f"          Decision:       {ev.decision.upper()}")
                print(f"          Confidence:     {conf}")
                print(f"          Vehicle Type:   {ev.vehicle_type}")
                print(f"          Transporter:    {ev.transporter}")
                print(f"          Timestamp:      {ev.timestamp}")
        else:
            print("  [-] No entry/exit events recorded for this plate.")

        # 3. OCR Plate Predictions & Multi-Frame Fusion
        print("\n[3] OCR Plate Predictions & AI Detections:")
        preds = db.query(PlatePrediction).filter(
            or_(
                PlatePrediction.raw_plate_text.like(f"%{clean_target}%"),
                PlatePrediction.clean_plate_text.like(f"%{clean_target}%"),
                PlatePrediction.fused_plate_text.like(f"%{clean_target}%"),
            )
        ).order_by(PlatePrediction.created_at.desc()).all()

        if preds:
            found_any = True
            for p in preds:
                conf = f"{p.confidence:.1%}" if p.confidence is not None else "N/A"
                print(f"  [FOUND] Prediction #{p.id}:")
                print(f"          Fused Plate:    {p.fused_plate_text}")
                print(f"          Clean Plate:    {p.clean_plate_text}")
                print(f"          Raw OCR Read:   {p.raw_plate_text}")
                print(f"          Confidence:     {conf}")
                print(f"          Frame Count:    {p.frame_count}")
                print(f"          Status:         {p.status}")
                print(f"          Detected At:    {p.created_at}")
        else:
            print("  [-] No OCR predictions recorded for this plate.")

        # 4. Scheduled Trips
        print("\n[4] Scheduled Trips:")
        trips = db.query(ScheduledTrip).filter(
            ScheduledTrip.plate_number.like(f"%{clean_target}%")
        ).all()

        if trips:
            found_any = True
            for t in trips:
                print(f"  [FOUND] Trip #{t.trip_number} (ID: {t.id}):")
                print(f"          Plate:          {t.plate_number}")
                print(f"          Driver:         {t.driver_name}")
                print(f"          Transporter:    {t.transporter_name}")
                print(f"          Purpose:        {t.purpose}")
                print(f"          Status:         {t.status}")
                print(f"          Expected:       {t.expected_arrival} to {t.expected_departure}")
        else:
            print("  [-] No scheduled trips found for this plate.")

        # 5. Whitelist / Watchlist
        print("\n[5] Security Access Lists:")
        wl = db.query(WhitelistEntry).filter(WhitelistEntry.plate_number.like(f"%{clean_target}%")).all()
        bl = db.query(WatchlistEntry).filter(WatchlistEntry.plate_number.like(f"%{clean_target}%")).all()

        if wl:
            found_any = True
            for w in wl:
                print(f"  [WHITELISTED] Reason: {w.reason} (Added: {w.created_at})")
        if bl:
            found_any = True
            for b in bl:
                print(f"  [WATCHLIST/BLACKLIST] Category: {b.category}, Reason: {b.reason} (Added: {b.created_at})")
        if not wl and not bl:
            print("  [-] Not present on Whitelist or Watchlist.")

        # 6. Alerts
        print("\n[6] Security Alerts:")
        alerts = db.query(Alert).filter(Alert.plate_number.like(f"%{clean_target}%")).all()
        if alerts:
            found_any = True
            for a in alerts:
                print(f"  [ALERT #{a.id}] Severity: {a.severity.upper()} | Type: {a.alert_type} | Message: {a.message} (Read: {a.is_read})")
        else:
            print("  [-] No security alerts for this plate.")

        # Summary
        print("\n" + "=" * 70)
        if found_any:
            print(f"  >>> RESULT: Vehicle '{target_plate}' IS STORED in the database! <<<")
        else:
            print(f"  >>> RESULT: Vehicle '{target_plate}' was NOT found in the database. <<<")
        print("=" * 70 + "\n")

    except Exception as exc:
        print(f"[-] Query error: {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    plate = sys.argv[1] if len(sys.argv) > 1 else "HR26FC2782"
    search_plate(plate)
