"""
SQLAlchemy Models for ANPRX Edge ANPR and Trip Management.
Contains all 24 production tables with proper keys, relationships, cascades, indexes, and constraints.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ─────────────────────────────────────────────────────────────
# 1. Master Data Models
# ─────────────────────────────────────────────────────────────

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, index=True, nullable=False)  # 'admin', 'manager', 'guard'
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    users = relationship("User", back_populates="role")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    role = relationship("Role", back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")


class Transporter(Base):
    __tablename__ = "transporters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    contact_person = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    email = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    vehicles = relationship("Vehicle", back_populates="transporter")
    drivers = relationship("Driver", back_populates="transporter")
    scheduled_trips = relationship("ScheduledTrip", back_populates="transporter")


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transporter_id = Column(Integer, ForeignKey("transporters.id", ondelete="SET NULL"), nullable=True, index=True)
    vehicle_type = Column(String(50), default="Truck", nullable=False)  # 'Car', 'Truck', 'Bus', 'Two wheeler'
    make = Column(String(50), nullable=True)
    model = Column(String(50), nullable=True)
    owner_name = Column(String(100), nullable=True)
    is_authorized = Column(Boolean, default=True, index=True, nullable=False)
    status = Column(String(50), default="Available", index=True, nullable=False)  # 'Available', 'Inside', 'On route', 'Scheduled', 'Flagged', 'Exited'
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    transporter = relationship("Transporter", back_populates="vehicles")
    plates = relationship("VehiclePlate", back_populates="vehicle", cascade="all, delete-orphan")
    drivers = relationship("Driver", back_populates="assigned_vehicle")
    trips = relationship("ScheduledTrip", back_populates="vehicle")
    events = relationship("EntryExitEvent", back_populates="vehicle")
    whitelist_entries = relationship("WhitelistEntry", back_populates="vehicle")
    watchlist_entries = relationship("WatchlistEntry", back_populates="vehicle")
    alerts = relationship("Alert", back_populates="vehicle")

    @property
    def primary_plate(self) -> str:
        if self.plates:
            for p in self.plates:
                if p.is_primary and p.is_active:
                    return p.plate_number
            return self.plates[0].plate_number
        return ""


class VehiclePlate(Base):
    __tablename__ = "vehicle_plates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    plate_number = Column(String(32), unique=True, index=True, nullable=False)
    is_primary = Column(Boolean, default=True, nullable=False)
    state_code = Column(String(10), nullable=True)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    vehicle = relationship("Vehicle", back_populates="plates")


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), index=True, nullable=False)
    license_number = Column(String(64), unique=True, index=True, nullable=False)
    phone = Column(String(50), nullable=False)
    transporter_id = Column(Integer, ForeignKey("transporters.id", ondelete="SET NULL"), nullable=True, index=True)
    assigned_vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True)
    status = Column(String(50), default="Available", index=True, nullable=False)  # 'Available', 'On site', 'Scheduled', 'Suspended'
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    transporter = relationship("Transporter", back_populates="drivers")
    assigned_vehicle = relationship("Vehicle", back_populates="drivers")
    trips = relationship("ScheduledTrip", back_populates="driver")


class Gate(Base):
    __tablename__ = "gates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, index=True, nullable=False)  # 'Gate 01', 'Gate 02', 'Gate 03'
    code = Column(String(20), unique=True, index=True, nullable=False)  # 'G01', 'G02', 'G03'
    location = Column(String(100), nullable=True)
    gate_type = Column(String(50), default="two_way", nullable=False)  # 'entry_only', 'exit_only', 'two_way'
    status = Column(String(50), default="active", index=True, nullable=False)  # 'active', 'maintenance', 'inactive'
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    cameras = relationship("Camera", back_populates="gate", cascade="all, delete-orphan")
    events = relationship("EntryExitEvent", back_populates="gate")
    daily_summaries = relationship("DailyGateSummary", back_populates="gate", cascade="all, delete-orphan")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), index=True, nullable=False)
    code = Column(String(50), unique=True, index=True, nullable=False)  # 'G01-ENTRY', 'G02-ENTRY', 'G03-EXIT', etc.
    rtsp_url = Column(String(255), nullable=True)
    ip_address = Column(String(50), nullable=True)
    direction = Column(String(20), default="entry", nullable=False)  # 'entry', 'exit', 'both'
    status = Column(String(50), default="online", index=True, nullable=False)  # 'online', 'offline', 'degraded'
    is_enabled = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    gate = relationship("Gate", back_populates="cameras")
    health_records = relationship("CameraHealth", back_populates="camera", cascade="all, delete-orphan")
    vehicle_detections = relationship("VehicleDetection", back_populates="camera")
    events = relationship("EntryExitEvent", back_populates="camera")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), index=True, nullable=False)  # 'yolov8n_vehicle', 'license_plate_yolo', 'pp_ocr_v4'
    version = Column(String(50), nullable=False)  # '8.4.116', '4.0.0'
    framework = Column(String(50), nullable=False)  # 'PyTorch', 'PaddlePaddle', 'TensorRT', 'ONNX'
    file_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    input_shape = Column(String(50), nullable=True)  # '640x640', 'dynamic'
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)


# ─────────────────────────────────────────────────────────────
# 2. Trip and Gate Operations Models
# ─────────────────────────────────────────────────────────────

class ScheduledTrip(Base):
    __tablename__ = "scheduled_trips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trip_number = Column(String(64), unique=True, index=True, nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True)
    plate_number = Column(String(32), index=True, nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id", ondelete="SET NULL"), nullable=True, index=True)
    driver_name = Column(String(100), nullable=True)
    transporter_id = Column(Integer, ForeignKey("transporters.id", ondelete="SET NULL"), nullable=True, index=True)
    transporter_name = Column(String(100), nullable=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="SET NULL"), nullable=True, index=True)
    gate_name = Column(String(50), nullable=True)
    purpose = Column(String(255), nullable=False)
    expected_arrival = Column(DateTime, index=True, nullable=False)
    expected_departure = Column(DateTime, nullable=False)
    actual_entry_time = Column(DateTime, nullable=True)
    actual_exit_time = Column(DateTime, nullable=True)
    dwell_minutes = Column(Integer, nullable=True)
    status = Column(String(50), default="scheduled", index=True, nullable=False)
    # Status lifecycle: 'scheduled' -> 'arrived' -> 'entry_approved' -> 'inside_plant' -> 'at_destination' -> 'exit_detected' -> 'completed', 'cancelled', 'exception'
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    vehicle = relationship("Vehicle", back_populates="trips")
    driver = relationship("Driver", back_populates="trips")
    transporter = relationship("Transporter", back_populates="scheduled_trips")
    status_history = relationship("TripStatusHistory", back_populates="trip", cascade="all, delete-orphan", order_by="TripStatusHistory.id.asc()")


class TripStatusHistory(Base):
    __tablename__ = "trip_status_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trip_id = Column(Integer, ForeignKey("scheduled_trips.id", ondelete="CASCADE"), nullable=False, index=True)
    from_status = Column(String(50), nullable=True)
    to_status = Column(String(50), index=True, nullable=False)
    changed_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    trigger_event_id = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)

    trip = relationship("ScheduledTrip", back_populates="status_history")


class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="SET NULL"), nullable=True, index=True)
    tracking_id = Column(Integer, index=True, nullable=False)
    frame_number = Column(Integer, nullable=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    vehicle_type = Column(String(50), default="car", nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    detection_timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    camera = relationship("Camera", back_populates="vehicle_detections")
    plate_predictions = relationship("PlatePrediction", back_populates="vehicle_detection")
    events = relationship("EntryExitEvent", back_populates="vehicle_detection")


class PlatePrediction(Base):
    __tablename__ = "plate_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_detection_id = Column(Integer, ForeignKey("vehicle_detections.id", ondelete="SET NULL"), nullable=True, index=True)
    track_id = Column(Integer, index=True, nullable=True)
    raw_plate_text = Column(String(64), nullable=True)
    clean_plate_text = Column(String(64), nullable=True)
    fused_plate_text = Column(String(64), index=True, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    is_fused = Column(Boolean, default=False, nullable=False)
    frame_count = Column(Integer, default=1, nullable=False)
    status = Column(String(50), default="finalized", index=True, nullable=False)  # 'finalized', 'manual_review', 'flagged'
    camera_code = Column(String(50), nullable=True)
    gate_name = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)

    vehicle_detection = relationship("VehicleDetection", back_populates="plate_predictions")
    frames = relationship("PlatePredictionFrame", back_populates="plate_prediction", cascade="all, delete-orphan")
    corrections = relationship("ManualCorrection", back_populates="plate_prediction")
    events = relationship("EntryExitEvent", back_populates="plate_prediction")


class PlatePredictionFrame(Base):
    __tablename__ = "plate_prediction_frames"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_prediction_id = Column(Integer, ForeignKey("plate_predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=True)
    ocr_text = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    crop_image_path = Column(String(255), nullable=True)
    engine = Column(String(50), default="PaddleOCR", nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    plate_prediction = relationship("PlatePrediction", back_populates="frames")


class EntryExitEvent(Base):
    __tablename__ = "entry_exit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True)
    vehicle_plate_id = Column(Integer, ForeignKey("vehicle_plates.id", ondelete="SET NULL"), nullable=True, index=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="SET NULL"), nullable=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True, index=True)
    vehicle_detection_id = Column(Integer, ForeignKey("vehicle_detections.id", ondelete="SET NULL"), nullable=True, index=True)
    plate_prediction_id = Column(Integer, ForeignKey("plate_predictions.id", ondelete="SET NULL"), nullable=True, index=True)
    event_type = Column(String(20), index=True, nullable=False)  # 'entry' | 'exit'
    plate_number = Column(String(32), index=True, nullable=False)
    vehicle_type = Column(String(50), default="Car", nullable=False)
    transporter = Column(String(100), default="Unregistered", nullable=False)
    gate_name = Column(String(50), default="Gate 01", nullable=False)
    camera_name = Column(String(100), default="G01-ENTRY", nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    decision = Column(String(30), default="allow", index=True, nullable=False)  # 'allow' | 'deny' | 'manual_review'
    is_corrected = Column(Boolean, default=False, nullable=False)
    dwell_minutes = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    vehicle = relationship("Vehicle", back_populates="events")
    gate = relationship("Gate", back_populates="events")
    camera = relationship("Camera", back_populates="events")
    vehicle_detection = relationship("VehicleDetection", back_populates="events")
    plate_prediction = relationship("PlatePrediction", back_populates="events")
    decisions = relationship("GateDecision", back_populates="event", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="event")


class GateDecision(Base):
    __tablename__ = "gate_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("entry_exit_events.id", ondelete="CASCADE"), nullable=False, index=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="SET NULL"), nullable=True, index=True)
    decision = Column(String(30), index=True, nullable=False)  # 'allow' | 'deny' | 'manual_review'
    reason = Column(String(255), nullable=True)
    rule_matched = Column(String(100), nullable=True)  # 'whitelist_match', 'watchlist_blacklist', 'low_confidence', 'scheduled_trip_valid'
    decided_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_override = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)

    event = relationship("EntryExitEvent", back_populates="decisions")


# ─────────────────────────────────────────────────────────────
# 3. Security Models
# ─────────────────────────────────────────────────────────────

class WhitelistEntry(Base):
    __tablename__ = "whitelist_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True)
    plate_number = Column(String(32), unique=True, index=True, nullable=False)
    reason = Column(String(255), nullable=True)
    approved_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    valid_from = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    valid_until = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    vehicle = relationship("Vehicle", back_populates="whitelist_entries")


class WatchlistEntry(Base):
    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True)
    plate_number = Column(String(32), unique=True, index=True, nullable=False)
    reason = Column(String(255), nullable=False)
    severity = Column(String(20), default="high", nullable=False)  # 'high', 'critical', 'medium'
    alert_message = Column(String(255), nullable=False)
    added_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    vehicle = relationship("Vehicle", back_populates="watchlist_entries")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), index=True, nullable=False)  # 'Unknown Vehicle', 'Watchlist Alert', 'Overstay', 'Plate Mismatch', 'Camera Offline'
    severity = Column(String(20), index=True, nullable=False)  # 'high', 'medium', 'low'
    message = Column(String(255), nullable=False)
    plate_number = Column(String(32), index=True, nullable=False)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="SET NULL"), nullable=True, index=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="SET NULL"), nullable=True, index=True)
    gate_name = Column(String(50), default="Gate 01", nullable=False)
    event_id = Column(Integer, ForeignKey("entry_exit_events.id", ondelete="SET NULL"), nullable=True, index=True)
    is_read = Column(Boolean, default=False, index=True, nullable=False)
    read_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)

    vehicle = relationship("Vehicle", back_populates="alerts")
    event = relationship("EntryExitEvent", back_populates="alerts")
    deliveries = relationship("AlertDelivery", back_populates="alert", cascade="all, delete-orphan")


class AlertDelivery(Base):
    __tablename__ = "alert_deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey("alerts.id", ondelete="CASCADE"), nullable=False, index=True)
    channel = Column(String(50), default="websocket", nullable=False)  # 'websocket', 'sms', 'email', 'webhook'
    recipient = Column(String(100), default="guard_console", nullable=False)
    status = Column(String(30), default="delivered", nullable=False)
    sent_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    error_message = Column(String(255), nullable=True)

    alert = relationship("Alert", back_populates="deliveries")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    action = Column(String(100), index=True, nullable=False)  # 'LOGIN', 'MANUAL_CORRECTION', 'VEHICLE_CREATED', 'CAMERA_UPDATED', etc.
    entity_type = Column(String(50), index=True, nullable=False)  # 'USER', 'VEHICLE', 'PLATE_PREDICTION', 'CAMERA'
    entity_id = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)

    user = relationship("User", back_populates="audit_logs")


# ─────────────────────────────────────────────────────────────
# 4. Review and Monitoring Models
# ─────────────────────────────────────────────────────────────

class ManualCorrection(Base):
    __tablename__ = "manual_corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_prediction_id = Column(Integer, ForeignKey("plate_predictions.id", ondelete="SET NULL"), nullable=True, index=True)
    event_id = Column(Integer, ForeignKey("entry_exit_events.id", ondelete="SET NULL"), nullable=True, index=True)
    original_plate = Column(String(32), nullable=False)
    corrected_plate = Column(String(32), index=True, nullable=False)
    operator_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reason = Column(String(255), nullable=True)
    corrected_at = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)

    plate_prediction = relationship("PlatePrediction", back_populates="corrections")


class CameraHealth(Base):
    __tablename__ = "camera_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    is_online = Column(Boolean, default=True, index=True, nullable=False)
    last_successful_frame_at = Column(DateTime, nullable=True)
    last_check_at = Column(DateTime, default=datetime.datetime.utcnow, index=True, nullable=False)
    response_time_ms = Column(Float, default=0.0, nullable=False)
    error_message = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    camera = relationship("Camera", back_populates="health_records")


class DailyGateSummary(Base):
    __tablename__ = "daily_gate_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gate_id = Column(Integer, ForeignKey("gates.id", ondelete="CASCADE"), nullable=False, index=True)
    summary_date = Column(Date, index=True, nullable=False)
    total_entries = Column(Integer, default=0, nullable=False)
    total_exits = Column(Integer, default=0, nullable=False)
    total_denied = Column(Integer, default=0, nullable=False)
    total_reviews = Column(Integer, default=0, nullable=False)
    total_alerts = Column(Integer, default=0, nullable=False)
    avg_dwell_minutes = Column(Float, default=0.0, nullable=False)
    calculated_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    gate = relationship("Gate", back_populates="daily_summaries")

    __table_args__ = (
        Index("idx_gate_date", "gate_id", "summary_date", unique=True),
    )
