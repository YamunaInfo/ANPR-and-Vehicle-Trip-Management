"""
Production Database Service for ANPRX Edge ANPR Platform.
Connects AI Pipeline (YOLO detection, PaddleOCR/EasyOCR, Multi-frame fusion)
directly to MySQL 8.x production database tables via SQLAlchemy.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from db.models import (
    Alert,
    AlertDelivery,
    AuditLog,
    Camera,
    CameraHealth,
    DailyGateSummary,
    Driver,
    EntryExitEvent,
    Gate,
    GateDecision,
    ManualCorrection,
    PlatePrediction,
    PlatePredictionFrame,
    ScheduledTrip,
    Transporter,
    TripStatusHistory,
    User,
    Vehicle,
    VehicleDetection,
    VehiclePlate,
    WatchlistEntry,
    WhitelistEntry,
)
from db.session import SessionLocal, init_db

logger = logging.getLogger("anprx.db_service")

# Configurable deduplication window (seconds)
DEDUP_WINDOW_SECONDS = int(os.environ.get("DEDUPLICATION_WINDOW_SECONDS", "15"))


class AnprDatabaseService:
    def __init__(self):
        self._ensure_initialized()

    def _ensure_initialized(self):
        try:
            init_db()
        except Exception as e:
            print(f"[DB WARN] Initialization notice: {e}", file=sys.stderr)

    def get_session(self) -> Session:
        return SessionLocal()

    def lookup_vehicle(self, plate: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """
        Looks up a vehicle by plate number in MySQL.
        """
        clean = (plate or "").upper().replace(" ", "").replace("-", "")
        if not clean:
            return None

        should_close = False
        if db is None:
            db = self.get_session()
            should_close = True

        try:
            plate_rec = db.query(VehiclePlate).filter(
                func.replace(func.replace(VehiclePlate.plate_number, " ", ""), "-", "") == clean,
                VehiclePlate.is_active == True
            ).first()

            if plate_rec and plate_rec.vehicle:
                veh = plate_rec.vehicle
                trans_name = veh.transporter.name if veh.transporter else "Unregistered"
                return {
                    "id": veh.id,
                    "plate_number": plate_rec.plate_number,
                    "vehicle_type": veh.vehicle_type,
                    "owner_name": veh.owner_name,
                    "transporter": trans_name,
                    "is_authorized": veh.is_authorized,
                    "status": veh.status,
                }
            return None
        finally:
            if should_close:
                db.close()

    def lookup_scheduled_trip(self, plate: str, db: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """
        Looks up the active scheduled trip for a given vehicle plate.
        """
        clean = (plate or "").upper().replace(" ", "").replace("-", "")
        if not clean:
            return None

        should_close = False
        if db is None:
            db = self.get_session()
            should_close = True

        try:
            trip = db.query(ScheduledTrip).filter(
                func.replace(func.replace(ScheduledTrip.plate_number, " ", ""), "-", "") == clean,
                ScheduledTrip.status.notin_(["completed", "cancelled"])
            ).order_by(ScheduledTrip.id.desc()).first()

            if trip:
                return {
                    "id": trip.id,
                    "trip_number": trip.trip_number,
                    "plate": trip.plate_number,
                    "driver": trip.driver_name,
                    "transporter": trip.transporter_name,
                    "gate": trip.gate_name,
                    "purpose": trip.purpose,
                    "expected_arrival": trip.expected_arrival.isoformat() if trip.expected_arrival else None,
                    "expected_departure": trip.expected_departure.isoformat() if trip.expected_departure else None,
                    "status": trip.status,
                    "entry_time": trip.actual_entry_time.isoformat() if trip.actual_entry_time else None,
                    "dwell_minutes": trip.dwell_minutes,
                }
            return None
        finally:
            if should_close:
                db.close()

    def check_whitelist(self, plate: str, db: Optional[Session] = None) -> Optional[WhitelistEntry]:
        clean = (plate or "").upper().replace(" ", "").replace("-", "")
        should_close = False
        if db is None:
            db = self.get_session()
            should_close = True
        try:
            return db.query(WhitelistEntry).filter(
                func.replace(func.replace(WhitelistEntry.plate_number, " ", ""), "-", "") == clean,
                WhitelistEntry.is_active == True
            ).first()
        finally:
            if should_close:
                db.close()

    def check_watchlist(self, plate: str, db: Optional[Session] = None) -> Optional[WatchlistEntry]:
        clean = (plate or "").upper().replace(" ", "").replace("-", "")
        should_close = False
        if db is None:
            db = self.get_session()
            should_close = True
        try:
            return db.query(WatchlistEntry).filter(
                func.replace(func.replace(WatchlistEntry.plate_number, " ", ""), "-", "") == clean,
                WatchlistEntry.is_active == True
            ).first()
        finally:
            if should_close:
                db.close()

    def record_finalized_anpr_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a finalized multi-frame ANPR event into MySQL:
        1. Saves vehicle detection record (if tracking info present)
        2. Saves plate prediction and OCR frame evidence
        3. Checks deduplication window to avoid duplicate passage entries
        4. Matches vehicle master, whitelist, watchlist
        5. Computes automated gate decision
        6. Inserts Entry/Exit event and GateDecision
        7. Progresses trip lifecycle and logs TripStatusHistory
        8. Calculates dwell time on exit
        9. Creates database alerts for watchlist, unknown, or low-confidence reads
        """
        raw_plate = event_data.get("plate_number", "")
        clean_plate = "".join(c for c in (raw_plate or "").upper() if c.isalnum())
        status = event_data.get("status", "finalized")
        track_id = event_data.get("track_id", 0)
        confidence = float(event_data.get("final_confidence", 0.90))
        if confidence > 1.0:
            confidence = confidence / 100.0
        vehicle_type = (event_data.get("vehicle_type") or "Car").capitalize()
        camera_code = event_data.get("camera_id", "G01-ENTRY")
        gate_name = event_data.get("gate_id", "Gate 01")
        is_corrected = bool(event_data.get("is_corrected", False))
        supporting_preds = event_data.get("supporting_predictions", [])
        frame_preds = event_data.get("frame_predictions", [])
        bbox = event_data.get("vehicle_bbox") or [0, 0, 0, 0]

        now_utc = datetime.datetime.utcnow()

        db = self.get_session()
        try:
            # 1. Resolve Gate & Camera from DB
            gate_rec = db.query(Gate).filter(Gate.name == gate_name).first()
            if not gate_rec:
                gate_rec = db.query(Gate).filter(Gate.code == gate_name).first()

            cam_rec = db.query(Camera).filter(Camera.code == camera_code).first()
            if not cam_rec:
                cam_rec = db.query(Camera).filter(Camera.name == camera_code).first()

            gate_id = gate_rec.id if gate_rec else None
            camera_id = cam_rec.id if cam_rec else None
            direction = cam_rec.direction if cam_rec else ("exit" if "EXIT" in camera_code.upper() else "entry")
            event_type = "exit" if direction == "exit" or "EXIT" in camera_code.upper() else "entry"

            # 2. Record Vehicle Detection
            detection_rec = VehicleDetection(
                camera_id=camera_id,
                gate_id=gate_id,
                tracking_id=int(track_id) if track_id else 0,
                frame_number=event_data.get("frame_number", 1),
                bbox_x1=float(bbox[0]) if len(bbox) > 0 else 0.0,
                bbox_y1=float(bbox[1]) if len(bbox) > 1 else 0.0,
                bbox_x2=float(bbox[2]) if len(bbox) > 2 else 0.0,
                bbox_y2=float(bbox[3]) if len(bbox) > 3 else 0.0,
                vehicle_type=vehicle_type,
                confidence=confidence,
                detection_timestamp=now_utc,
            )
            db.add(detection_rec)
            db.flush()

            # 3. Record Plate Prediction & Individual Frame Evidence
            pred_rec = PlatePrediction(
                vehicle_detection_id=detection_rec.id,
                track_id=int(track_id) if track_id else 0,
                raw_plate_text=raw_plate,
                clean_plate_text=clean_plate,
                fused_plate_text=clean_plate if clean_plate else "UNKNOWN",
                confidence=confidence,
                is_fused=len(supporting_preds) > 1 or len(frame_preds) > 1,
                frame_count=max(len(frame_preds), event_data.get("frame_count", 1)),
                status=status if confidence >= 0.75 else "manual_review",
                camera_code=camera_code,
                gate_name=gate_name,
                created_at=now_utc,
            )
            db.add(pred_rec)
            db.flush()

            # Save individual OCR frame predictions into plate_prediction_frames
            if frame_preds:
                for fp in frame_preds:
                    f_num = fp.get("frame_number", 0)
                    f_text = fp.get("plate", fp.get("text", ""))
                    f_conf = float(fp.get("confidence", 0.0))
                    f_eng = fp.get("engine", "PaddleOCR")
                    db.add(PlatePredictionFrame(
                        plate_prediction_id=pred_rec.id,
                        frame_number=f_num,
                        ocr_text=f_text,
                        confidence=f_conf,
                        engine=f_eng,
                        timestamp=now_utc,
                    ))
            elif clean_plate:
                db.add(PlatePredictionFrame(
                    plate_prediction_id=pred_rec.id,
                    frame_number=1,
                    ocr_text=clean_plate,
                    confidence=confidence,
                    engine="PaddleOCR",
                    timestamp=now_utc,
                ))

            # 4. Check Deduplication
            # If the same plate was registered in the same gate/direction within DEDUP_WINDOW_SECONDS, skip creating duplicate entry/exit event
            if clean_plate and clean_plate != "UNKNOWN":
                cutoff_time = now_utc - datetime.timedelta(seconds=DEDUP_WINDOW_SECONDS)
                recent_event = db.query(EntryExitEvent).filter(
                    EntryExitEvent.plate_number == clean_plate,
                    EntryExitEvent.event_type == event_type,
                    EntryExitEvent.timestamp >= cutoff_time,
                ).order_by(EntryExitEvent.id.desc()).first()

                if recent_event:
                    db.commit()
                    return {
                        "status": "duplicate_skipped",
                        "decision": recent_event.decision,
                        "plate": clean_plate,
                        "event_id": recent_event.id,
                        "dwell_minutes": recent_event.dwell_minutes,
                    }

            # 5. Master Data Lookup & Security Checks
            vehicle_rec = None
            vehicle_plate_rec = None
            if clean_plate:
                vehicle_plate_rec = db.query(VehiclePlate).filter(
                    func.replace(func.replace(VehiclePlate.plate_number, " ", ""), "-", "") == clean_plate
                ).first()
                if vehicle_plate_rec:
                    vehicle_rec = vehicle_plate_rec.vehicle

            whitelist_match = self.check_whitelist(clean_plate, db=db)
            watchlist_match = self.check_watchlist(clean_plate, db=db)

            # 6. Gate Decision Logic
            decision = "allow"
            decision_reason = "Authorized entry"
            rule_matched = "standard_access"

            if confidence < 0.75:
                decision = "manual_review"
                decision_reason = f"Low OCR recognition confidence ({int(confidence * 100)}% < 75%)"
                rule_matched = "low_confidence_threshold"
            elif watchlist_match or (vehicle_rec and not vehicle_rec.is_authorized):
                decision = "deny"
                decision_reason = watchlist_match.reason if watchlist_match else "Vehicle marked as Flagged/Unauthorized"
                rule_matched = "watchlist_blacklist_enforcement"
            elif whitelist_match:
                decision = "allow"
                decision_reason = "Vehicle matched valid Whitelist registration"
                rule_matched = "whitelist_preapproved"
            elif vehicle_rec:
                decision = "allow" if vehicle_rec.is_authorized else "deny"
                decision_reason = "Registered authorized vehicle" if vehicle_rec.is_authorized else "Unauthorized vehicle"
                rule_matched = "master_vehicle_registry"
            else:
                # Unknown vehicle
                decision = "manual_review"
                decision_reason = "Unregistered vehicle not found in database"
                rule_matched = "unknown_vehicle_review"

            # 7. Resolve Transporter & Details
            transporter_name = (
                vehicle_rec.transporter.name
                if (vehicle_rec and vehicle_rec.transporter)
                else "Unregistered"
            )

            # 8. Trip Management & Dwell Time Calculation
            matched_trip = None
            dwell_minutes = None

            if clean_plate and clean_plate != "UNKNOWN":
                matched_trip = db.query(ScheduledTrip).filter(
                    func.replace(func.replace(ScheduledTrip.plate_number, " ", ""), "-", "") == clean_plate,
                    ScheduledTrip.status.notin_(["completed", "cancelled"])
                ).order_by(ScheduledTrip.id.desc()).first()

            if event_type == "entry":
                if matched_trip and decision == "allow":
                    from_status = matched_trip.status
                    matched_trip.status = "inside_plant"
                    matched_trip.actual_entry_time = now_utc
                    matched_trip.dwell_minutes = None
                    db.add(TripStatusHistory(
                        trip_id=matched_trip.id,
                        from_status=from_status,
                        to_status="inside_plant",
                        trigger_event_id=detection_rec.id,
                        notes=f"ANPR Auto-Entry at {gate_name} via {camera_code}",
                        timestamp=now_utc,
                    ))
                elif not matched_trip and decision == "allow":
                    # Create automatic operational trip
                    matched_trip = ScheduledTrip(
                        trip_number=f"TRIP-AUTO-{int(time.time()) % 100000}",
                        vehicle_id=vehicle_rec.id if vehicle_rec else None,
                        plate_number=clean_plate,
                        driver_name=vehicle_rec.owner_name if vehicle_rec else "Driver",
                        transporter_id=vehicle_rec.transporter_id if vehicle_rec else None,
                        transporter_name=transporter_name,
                        gate_id=gate_id,
                        gate_name=gate_name,
                        purpose="Live ANPR Gate Entry",
                        expected_arrival=now_utc,
                        expected_departure=now_utc + datetime.timedelta(hours=2),
                        actual_entry_time=now_utc,
                        status="inside_plant",
                        notes="Auto-created by ANPR live detection",
                    )
                    db.add(matched_trip)
                    db.flush()
                    db.add(TripStatusHistory(
                        trip_id=matched_trip.id,
                        from_status="scheduled",
                        to_status="inside_plant",
                        trigger_event_id=detection_rec.id,
                        notes=f"Ad-hoc ANPR Entry at {gate_name}",
                        timestamp=now_utc,
                    ))

                if vehicle_rec:
                    vehicle_rec.status = "Inside"

            elif event_type == "exit":
                if matched_trip:
                    from_status = matched_trip.status
                    matched_trip.status = "completed"
                    matched_trip.actual_exit_time = now_utc
                    if matched_trip.actual_entry_time:
                        delta = (now_utc - matched_trip.actual_entry_time).total_seconds() / 60.0
                        dwell_minutes = max(1, int(round(delta)))
                        matched_trip.dwell_minutes = dwell_minutes
                    else:
                        dwell_minutes = 25
                        matched_trip.dwell_minutes = 25

                    db.add(TripStatusHistory(
                        trip_id=matched_trip.id,
                        from_status=from_status,
                        to_status="completed",
                        trigger_event_id=detection_rec.id,
                        notes=f"ANPR Auto-Exit at {gate_name} via {camera_code}. Dwell: {dwell_minutes} mins",
                        timestamp=now_utc,
                    ))
                else:
                    dwell_minutes = 30

                if vehicle_rec:
                    vehicle_rec.status = "Exited"

            # 9. Create EntryExitEvent Record
            event_rec = EntryExitEvent(
                vehicle_id=vehicle_rec.id if vehicle_rec else None,
                vehicle_plate_id=vehicle_plate_rec.id if vehicle_plate_rec else None,
                gate_id=gate_id,
                camera_id=camera_id,
                vehicle_detection_id=detection_rec.id,
                plate_prediction_id=pred_rec.id,
                event_type=event_type,
                plate_number=clean_plate if clean_plate else "UNKNOWN",
                vehicle_type=vehicle_type,
                transporter=transporter_name,
                gate_name=gate_name,
                camera_name=camera_code,
                confidence=confidence,
                decision=decision,
                is_corrected=is_corrected,
                dwell_minutes=dwell_minutes,
                timestamp=now_utc,
                created_at=now_utc,
            )
            db.add(event_rec)
            db.flush()

            # 10. Create GateDecision Record
            db.add(GateDecision(
                event_id=event_rec.id,
                gate_id=gate_id,
                decision=decision,
                reason=decision_reason,
                rule_matched=rule_matched,
                timestamp=now_utc,
            ))

            # 11. Security Alerts Dispatch
            created_alert = None
            if watchlist_match or (vehicle_rec and not vehicle_rec.is_authorized):
                created_alert = Alert(
                    alert_type="Watchlist Alert",
                    severity="high",
                    message=f"⛔ Blacklisted / Watchlist vehicle {clean_plate} detected at {gate_name} - Access Denied!",
                    plate_number=clean_plate,
                    vehicle_id=vehicle_rec.id if vehicle_rec else None,
                    gate_id=gate_id,
                    gate_name=gate_name,
                    event_id=event_rec.id,
                    is_read=False,
                    created_at=now_utc,
                )
                db.add(created_alert)
                db.flush()
                db.add(AlertDelivery(
                    alert_id=created_alert.id,
                    channel="websocket",
                    recipient="security_guard_console",
                    status="delivered",
                    sent_at=now_utc,
                ))
            elif vehicle_rec is None and clean_plate and clean_plate != "UNKNOWN":
                created_alert = Alert(
                    alert_type="Unknown Vehicle",
                    severity="high",
                    message="Unregistered vehicle",
                    plate_number=clean_plate,
                    gate_id=gate_id,
                    gate_name=gate_name,
                    event_id=event_rec.id,
                    is_read=False,
                    created_at=now_utc,
                )
                db.add(created_alert)
                db.flush()
                db.add(AlertDelivery(
                    alert_id=created_alert.id,
                    channel="websocket",
                    recipient="guard_console",
                    status="delivered",
                    sent_at=now_utc,
                ))
            elif confidence < 0.75 and clean_plate:
                created_alert = Alert(
                    alert_type="Low Confidence Read",
                    severity="medium",
                    message=f"⚠️ Low confidence plate recognition for {clean_plate} ({int(confidence * 100)}% < 75%) at {gate_name}",
                    plate_number=clean_plate,
                    gate_id=gate_id,
                    gate_name=gate_name,
                    event_id=event_rec.id,
                    is_read=False,
                    created_at=now_utc,
                )
                db.add(created_alert)

            db.commit()

            return {
                "decision": decision,
                "status": "finalized",
                "plate": clean_plate,
                "event_id": event_rec.id,
                "transporter": transporter_name,
                "event_type": event_type,
                "dwell_minutes": dwell_minutes,
                "alert": {
                    "id": created_alert.id,
                    "type": created_alert.alert_type,
                    "message": created_alert.message,
                } if created_alert else None,
            }

        except Exception as e:
            db.rollback()
            print(f"[DB ERROR] Error recording ANPR event: {e}", file=sys.stderr)
            raise
        finally:
            db.close()


# Global persistent database service instance
db_service = AnprDatabaseService()
