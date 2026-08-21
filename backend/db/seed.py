"""
Idempotent Master Data Seed Script for ANPRX Edge ANPR Platform.
Seeds initial master records (Roles, Users, Transporters, Gates, Cameras, Vehicles, Drivers, Whitelist, Watchlist, Models, Scheduled Trips).
Does NOT continuously seed fake operational events (detections, predictions, frames, events, decisions, alerts, camera health).
Safe to run repeatedly without creating duplicates.
"""
from __future__ import annotations

import datetime
import hashlib
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import (
    Camera,
    Driver,
    Gate,
    ModelVersion,
    Role,
    ScheduledTrip,
    Transporter,
    User,
    Vehicle,
    VehiclePlate,
    WatchlistEntry,
    WhitelistEntry,
)
from db.session import SessionLocal, init_db


def hash_password(password: str) -> str:
    """Simple deterministic sha256 password hash for standard demo users."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def seed_master_data(db: Optional[Session] = None) -> None:
    should_close = False
    if db is None:
        init_db()
        db = SessionLocal()
        should_close = True

    try:
        print("[Seed] Starting idempotent master data seeding for ANPRX...")

        # 1. Roles
        roles_data = [
            {"name": "admin", "description": "System Administrator with full management access"},
            {"name": "manager", "description": "Operations & Logistics Manager"},
            {"name": "guard", "description": "Gate Security Guard Operator"},
        ]
        roles_map: Dict[str, Role] = {}
        for r_dict in roles_data:
            role = db.query(Role).filter_by(name=r_dict["name"]).first()
            if not role:
                role = Role(name=r_dict["name"], description=r_dict["description"])
                db.add(role)
                db.flush()
                print(f"[Seed] Created role: {role.name}")
            roles_map[role.name] = role

        # 2. Users
        default_pwd_hash = hash_password("password123")
        users_data = [
            {"name": "Priya Nair", "email": "admin@anprx.io", "role": "admin"},
            {"name": "Priya Nair", "email": "admin@gatesense.io", "role": "admin"},
            {"name": "Aarav Menon", "email": "manager@anprx.io", "role": "manager"},
            {"name": "Aarav Menon", "email": "manager@gatesense.io", "role": "manager"},
            {"name": "Ravi Kumar", "email": "guard@anprx.io", "role": "guard"},
            {"name": "Ravi Kumar", "email": "guard@gatesense.io", "role": "guard"},
        ]
        for u_dict in users_data:
            user = db.query(User).filter_by(email=u_dict["email"]).first()
            if not user:
                user = User(
                    name=u_dict["name"],
                    email=u_dict["email"],
                    password_hash=default_pwd_hash,
                    role_id=roles_map[u_dict["role"]].id,
                    is_active=True,
                )
                db.add(user)
                db.flush()
                print(f"[Seed] Created user: {user.email} ({u_dict['role']})")

        admin_user = db.query(User).filter_by(email="admin@anprx.io").first()

        # 3. Transporters
        transporters_data = [
            {"name": "BlueDart Logistics", "contact_person": "Rajesh Verma", "phone": "+91 98110 22334", "email": "dispatch@bluedart.com"},
            {"name": "Rana Freight", "contact_person": "Gurpreet Rana", "phone": "+91 98220 33445", "email": "ops@ranafreight.com"},
            {"name": "Eastline Carriers", "contact_person": "Bikram Das", "phone": "+91 98330 44556", "email": "support@eastline.in"},
            {"name": "Apex Haulage", "contact_person": "S. Ramanathan", "phone": "+91 98440 55667", "email": "transport@apexhaulage.in"},
        ]
        transporters_map: Dict[str, Transporter] = {}
        for t_dict in transporters_data:
            trans = db.query(Transporter).filter_by(name=t_dict["name"]).first()
            if not trans:
                trans = Transporter(
                    name=t_dict["name"],
                    contact_person=t_dict["contact_person"],
                    phone=t_dict["phone"],
                    email=t_dict["email"],
                    is_active=True,
                )
                db.add(trans)
                db.flush()
                print(f"[Seed] Created transporter: {trans.name}")
            transporters_map[trans.name] = trans

        # 4. Gates
        gates_data = [
            {"name": "Gate 01", "code": "G01", "location": "North Inbound/Outbound Complex", "gate_type": "two_way", "status": "active"},
            {"name": "Gate 02", "code": "G02", "location": "Loading Bay Logistics Facility", "gate_type": "two_way", "status": "active"},
            {"name": "Gate 03", "code": "G03", "location": "South Perimeter Express Exit", "gate_type": "exit_only", "status": "active"},
        ]
        gates_map: Dict[str, Gate] = {}
        for g_dict in gates_data:
            gate = db.query(Gate).filter_by(code=g_dict["code"]).first()
            if not gate:
                gate = Gate(
                    name=g_dict["name"],
                    code=g_dict["code"],
                    location=g_dict["location"],
                    gate_type=g_dict["gate_type"],
                    status=g_dict["status"],
                )
                db.add(gate)
                db.flush()
                print(f"[Seed] Created gate: {gate.name} ({gate.code})")
            gates_map[gate.name] = gate

        # 5. Cameras
        cameras_data = [
            {
                "name": "Main Entry Inbound",
                "code": "G01-ENTRY",
                "gate_name": "Gate 01",
                "direction": "entry",
                "status": "online",
                "rtsp_url": "rtsp://admin:password@192.168.1.64:554/stream1",
                "ip_address": "192.168.1.64",
            },
            {
                "name": "Loading Bay Inbound",
                "code": "G02-ENTRY",
                "gate_name": "Gate 02",
                "direction": "entry",
                "status": "online",
                "rtsp_url": "rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1",
                "ip_address": "192.168.1.108",
            },
            {
                "name": "Exit Lane",
                "code": "G03-EXIT",
                "gate_name": "Gate 03",
                "direction": "exit",
                "status": "online",
                "rtsp_url": "rtsp://admin:password@192.168.1.112:554/stream1",
                "ip_address": "192.168.1.112",
            },
            {
                "name": "Loading Bay Outbound",
                "code": "G02-EXIT",
                "gate_name": "Gate 02",
                "direction": "exit",
                "status": "online",
                "rtsp_url": "rtsp://admin:password@192.168.1.115:554/stream1",
                "ip_address": "192.168.1.115",
            },
        ]
        for c_dict in cameras_data:
            cam = db.query(Camera).filter_by(code=c_dict["code"]).first()
            if not cam:
                cam = Camera(
                    name=c_dict["name"],
                    code=c_dict["code"],
                    gate_id=gates_map[c_dict["gate_name"]].id,
                    direction=c_dict["direction"],
                    status=c_dict["status"],
                    rtsp_url=c_dict["rtsp_url"],
                    ip_address=c_dict["ip_address"],
                    is_enabled=True,
                )
                db.add(cam)
                db.flush()
                print(f"[Seed] Created camera: {cam.name} ({cam.code})")

        # 6. Vehicles & Plates
        vehicles_data = [
            {"plate": "TN37AB1234", "type": "Truck", "owner": "Arun Exports", "transporter": "BlueDart Logistics", "authorized": True, "status": "Inside"},
            {"plate": "KA05MZ5678", "type": "Truck", "owner": "Nandi Foods", "transporter": "Rana Freight", "authorized": True, "status": "Inside"},
            {"plate": "MH12QX9031", "type": "Car", "owner": "N. Rao", "transporter": "Eastline Carriers", "authorized": True, "status": "Available"},
            {"plate": "AP09TC4412", "type": "Bus", "owner": "Apex Staffing", "transporter": "Apex Haulage", "authorized": True, "status": "Scheduled"},
            {"plate": "GJ18BR2290", "type": "Truck", "owner": "Gujarat Steel", "transporter": "Rana Freight", "authorized": False, "status": "Flagged"},
            {"plate": "DL01LK8402", "type": "Car", "owner": "M. Kapoor", "transporter": "Eastline Carriers", "authorized": True, "status": "Exited"},
            {"plate": "WB12AB1234", "type": "Truck", "owner": "West Bengal Mills", "transporter": "BlueDart Logistics", "authorized": True, "status": "Scheduled"},
            {"plate": "TS08HN1922", "type": "Two wheeler", "owner": "S. Harish", "transporter": "Apex Haulage", "authorized": True, "status": "Exited"},
        ]
        vehicles_map: Dict[str, Vehicle] = {}
        for v_dict in vehicles_data:
            plate_rec = db.query(VehiclePlate).filter_by(plate_number=v_dict["plate"]).first()
            if not plate_rec:
                veh = Vehicle(
                    vehicle_type=v_dict["type"],
                    owner_name=v_dict["owner"],
                    transporter_id=transporters_map[v_dict["transporter"]].id if v_dict["transporter"] in transporters_map else None,
                    is_authorized=v_dict["authorized"],
                    status=v_dict["status"],
                )
                db.add(veh)
                db.flush()

                plate_rec = VehiclePlate(
                    vehicle_id=veh.id,
                    plate_number=v_dict["plate"],
                    is_primary=True,
                    state_code=v_dict["plate"][:2],
                    is_active=True,
                )
                db.add(plate_rec)
                db.flush()
                print(f"[Seed] Created vehicle & plate: {v_dict['plate']}")
                vehicles_map[v_dict["plate"]] = veh
            else:
                vehicles_map[v_dict["plate"]] = plate_rec.vehicle

        # 7. Drivers
        drivers_data = [
            {"name": "Suresh Kumar", "license": "TN-09-2018-004391", "phone": "+91 98402 11428", "plate": "TN37AB1234", "transporter": "BlueDart Logistics", "status": "On site"},
            {"name": "Mahesh Reddy", "license": "KA-05-2020-008221", "phone": "+91 99861 64012", "plate": "KA05MZ5678", "transporter": "Rana Freight", "status": "On site"},
            {"name": "Pradeep Singh", "license": "MH-12-2019-006782", "phone": "+91 98209 73114", "plate": "MH12QX9031", "transporter": "Eastline Carriers", "status": "Available"},
            {"name": "Anjali Nair", "license": "AP-09-2021-000837", "phone": "+91 98472 21930", "plate": "AP09TC4412", "transporter": "Apex Haulage", "status": "Scheduled"},
            {"name": "Vikram Shah", "license": "GJ-18-2017-004620", "phone": "+91 98980 99218", "plate": "GJ18BR2290", "transporter": "Rana Freight", "status": "Suspended"},
        ]
        drivers_map: Dict[str, Driver] = {}
        for d_dict in drivers_data:
            drv = db.query(Driver).filter_by(license_number=d_dict["license"]).first()
            if not drv:
                veh = vehicles_map.get(d_dict["plate"])
                drv = Driver(
                    name=d_dict["name"],
                    license_number=d_dict["license"],
                    phone=d_dict["phone"],
                    transporter_id=transporters_map.get(d_dict["transporter"]).id if d_dict["transporter"] in transporters_map else None,
                    assigned_vehicle_id=veh.id if veh else None,
                    status=d_dict["status"],
                    is_active=True,
                )
                db.add(drv)
                db.flush()
                print(f"[Seed] Created driver: {drv.name}")
            drivers_map[drv.name] = drv

        # 8. Whitelist Entries
        whitelist_plates = ["TN37AB1234", "KA05MZ5678", "MH12QX9031", "AP09TC4412", "DL01LK8402", "WB12AB1234", "TS08HN1922"]
        for p in whitelist_plates:
            w_entry = db.query(WhitelistEntry).filter_by(plate_number=p).first()
            if not w_entry:
                veh = vehicles_map.get(p)
                w_entry = WhitelistEntry(
                    vehicle_id=veh.id if veh else None,
                    plate_number=p,
                    reason="Authorized fleet regular operations",
                    approved_by_user_id=admin_user.id if admin_user else None,
                    valid_from=datetime.datetime.utcnow(),
                    is_active=True,
                )
                db.add(w_entry)
                print(f"[Seed] Added whitelist entry: {p}")

        # 9. Watchlist Entries (Flagged / Blacklisted)
        watchlist_data = [
            {"plate": "GJ18BR2290", "reason": "Suspended registration & repeated safety violations", "severity": "high", "message": "Watchlist vehicle denied entry at security perimeter"},
        ]
        for w_dict in watchlist_data:
            watch_entry = db.query(WatchlistEntry).filter_by(plate_number=w_dict["plate"]).first()
            if not watch_entry:
                veh = vehicles_map.get(w_dict["plate"])
                watch_entry = WatchlistEntry(
                    vehicle_id=veh.id if veh else None,
                    plate_number=w_dict["plate"],
                    reason=w_dict["reason"],
                    severity=w_dict["severity"],
                    alert_message=w_dict["message"],
                    added_by_user_id=admin_user.id if admin_user else None,
                    is_active=True,
                )
                db.add(watch_entry)
                print(f"[Seed] Added watchlist entry: {w_dict['plate']}")

        # 10. Model Versions (Metadata only, model files stored separately)
        models_data = [
            {"model_name": "yolov8n_vehicle", "version": "8.4.116", "framework": "PyTorch", "file_path": "models/yolov8n.pt", "input_shape": "640x640"},
            {"model_name": "license_plate_yolo", "version": "8.4.116", "framework": "PyTorch", "file_path": "models/license_plate.pt", "input_shape": "640x640"},
            {"model_name": "paddleocr_ppocrv4", "version": "4.0.0", "framework": "PaddlePaddle", "file_path": "PaddleOCR/PP-OCRv4", "input_shape": "dynamic"},
        ]
        for m_dict in models_data:
            m_rec = db.query(ModelVersion).filter_by(model_name=m_dict["model_name"]).first()
            if not m_rec:
                m_rec = ModelVersion(
                    model_name=m_dict["model_name"],
                    version=m_dict["version"],
                    framework=m_dict["framework"],
                    file_path=m_dict["file_path"],
                    input_shape=m_dict["input_shape"],
                    is_active=True,
                )
                db.add(m_rec)
                print(f"[Seed] Registered model metadata: {m_rec.model_name}")

        # 11. Initial Scheduled Trips (Master Data)
        now_dt = datetime.datetime.utcnow()
        trips_seed = [
            {
                "trip_number": "TRIP-701",
                "plate": "TN37AB1234",
                "driver": "Suresh Kumar",
                "transporter": "BlueDart Logistics",
                "gate": "Gate 01",
                "purpose": "Raw material delivery",
                "expected_arrival": now_dt - datetime.timedelta(minutes=62),
                "expected_departure": now_dt + datetime.timedelta(minutes=90),
                "status": "inside_plant",
                "actual_entry_time": now_dt - datetime.timedelta(minutes=46),
                "dwell_minutes": None,
            },
            {
                "trip_number": "TRIP-702",
                "plate": "KA05MZ5678",
                "driver": "Mahesh Reddy",
                "transporter": "Rana Freight",
                "gate": "Gate 02",
                "purpose": "Finished goods pickup",
                "expected_arrival": now_dt - datetime.timedelta(minutes=33),
                "expected_departure": now_dt + datetime.timedelta(minutes=42),
                "status": "at_destination",
                "actual_entry_time": now_dt - datetime.timedelta(minutes=28),
                "dwell_minutes": None,
            },
            {
                "trip_number": "TRIP-703",
                "plate": "MH12QX9031",
                "driver": "Pradeep Singh",
                "transporter": "Eastline Carriers",
                "gate": "Gate 01",
                "purpose": "Vendor inspection",
                "expected_arrival": now_dt + datetime.timedelta(minutes=28),
                "expected_departure": now_dt + datetime.timedelta(minutes=140),
                "status": "scheduled",
                "actual_entry_time": None,
                "dwell_minutes": None,
            },
            {
                "trip_number": "TRIP-704",
                "plate": "AP09TC4412",
                "driver": "Anjali Nair",
                "transporter": "Apex Haulage",
                "gate": "Gate 03",
                "purpose": "Shift transport",
                "expected_arrival": now_dt + datetime.timedelta(minutes=13),
                "expected_departure": now_dt + datetime.timedelta(minutes=100),
                "status": "scheduled",
                "actual_entry_time": None,
                "dwell_minutes": None,
            },
            {
                "trip_number": "TRIP-705",
                "plate": "WB12AB1234",
                "driver": "Suresh Kumar",
                "transporter": "BlueDart Logistics",
                "gate": "Gate 01",
                "purpose": "Component delivery",
                "expected_arrival": now_dt + datetime.timedelta(minutes=45),
                "expected_departure": now_dt + datetime.timedelta(minutes=165),
                "status": "scheduled",
                "actual_entry_time": None,
                "dwell_minutes": None,
            },
        ]
        for t_dict in trips_seed:
            t_rec = db.query(ScheduledTrip).filter_by(trip_number=t_dict["trip_number"]).first()
            if not t_rec:
                veh = vehicles_map.get(t_dict["plate"])
                drv = drivers_map.get(t_dict["driver"])
                trans = transporters_map.get(t_dict["transporter"])
                gate = gates_map.get(t_dict["gate"])
                t_rec = ScheduledTrip(
                    trip_number=t_dict["trip_number"],
                    vehicle_id=veh.id if veh else None,
                    plate_number=t_dict["plate"],
                    driver_id=drv.id if drv else None,
                    driver_name=t_dict["driver"],
                    transporter_id=trans.id if trans else None,
                    transporter_name=t_dict["transporter"],
                    gate_id=gate.id if gate else None,
                    gate_name=t_dict["gate"],
                    purpose=t_dict["purpose"],
                    expected_arrival=t_dict["expected_arrival"],
                    expected_departure=t_dict["expected_departure"],
                    actual_entry_time=t_dict["actual_entry_time"],
                    dwell_minutes=t_dict["dwell_minutes"],
                    status=t_dict["status"],
                )
                db.add(t_rec)
                print(f"[Seed] Created scheduled trip: {t_dict['trip_number']} for {t_dict['plate']}")

        db.commit()
        print("[Seed] Idempotent master data seeding complete!")

    except Exception as exc:
        db.rollback()
        print(f"[Seed ERROR] Seeding failed: {exc}")
        raise
    finally:
        if should_close:
            db.close()


if __name__ == "__main__":
    seed_master_data()
