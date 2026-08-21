"""
Comprehensive test suite verifying the unified Python FastAPI backend.
Tests all REST endpoints migrated from Express alongside existing AI/OCR endpoints.
"""
import sys
import unittest
from fastapi.testclient import TestClient

from ocr_service import app

client = TestClient(app)


class TestUnifiedBackend(unittest.TestCase):

    def test_01_health_endpoints(self):
        # Root health check
        r1 = client.get("/healthz")
        self.assertEqual(r1.status_code, 200)
        self.assertIn("status", r1.json())

        # API health check
        r2 = client.get("/api/healthz")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json().get("status"), "ok")

        # AI health check
        r3 = client.get("/api/ai/health")
        self.assertEqual(r3.status_code, 200)
        self.assertIn("status", r3.json())

    def test_02_auth_flow(self):
        # Register new user
        reg_payload = {
            "name": "Test Inspector",
            "email": "inspector@gatesense.io",
            "password": "secretpassword",
            "role": "guard"
        }
        r1 = client.post("/api/auth/register", json=reg_payload)
        self.assertEqual(r1.status_code, 200)
        data = r1.json()
        self.assertIn("token", data)
        self.assertEqual(data["operator"]["email"], "inspector@gatesense.io")

        # Login existing user
        login_payload = {
            "email": "inspector@gatesense.io",
            "password": "secretpassword"
        }
        r2 = client.post("/api/auth/login", json=login_payload)
        self.assertEqual(r2.status_code, 200)
        self.assertIn("token", r2.json())

        # Current operator profile
        r3 = client.get("/api/me", headers={"Authorization": "Bearer session-admin-123"})
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(r3.json().get("role"), "admin")

    def test_03_dashboard_endpoints(self):
        r_sum = client.get("/api/dashboard/summary")
        self.assertEqual(r_sum.status_code, 200)
        sum_data = r_sum.json()
        for key in ["vehiclesInside", "entriesToday", "exitsToday", "activeAlerts", "avgDwellMinutes", "recognitionAccuracy", "gatesOnline", "totalGates"]:
            self.assertIn(key, sum_data)

        r_act = client.get("/api/dashboard/activity")
        self.assertEqual(r_act.status_code, 200)
        self.assertIsInstance(r_act.json(), list)

    def test_04_events_and_simulation(self):
        # List events
        r_ev = client.get("/api/events")
        self.assertEqual(r_ev.status_code, 200)
        self.assertIsInstance(r_ev.json(), list)

        # Simulate a gate event
        r_sim = client.post("/api/events/simulate")
        self.assertEqual(r_sim.status_code, 201)
        sim_data = r_sim.json()
        self.assertIn("event", sim_data)
        self.assertIn("plate", sim_data["event"])

        # Detection consensus endpoint
        r_det = client.post("/api/detections", json={"frames": ["TN37AB1234", "TN37A81234"], "vehicleType": "Truck"})
        self.assertEqual(r_det.status_code, 201)
        det_data = r_det.json()
        self.assertEqual(det_data["finalPlate"], "TN37AB1234")

    def test_05_trips_management(self):
        # Get trips
        r1 = client.get("/api/trips")
        self.assertEqual(r1.status_code, 200)

        # Active trips
        r2 = client.get("/api/trips/active")
        self.assertEqual(r2.status_code, 200)

        # Create new trip
        new_trip = {
            "plate": "MH12QX9031",
            "driver": "Pradeep Singh",
            "transporter": "Eastline Carriers",
            "gate": "Gate 01",
            "purpose": "Component Inspection",
            "expectedArrival": "2026-08-21T10:00:00Z",
            "expectedDeparture": "2026-08-21T12:00:00Z"
        }
        r3 = client.post("/api/trips", json=new_trip)
        self.assertEqual(r3.status_code, 201)
        trip_id = r3.json()["id"]

        # Update status
        r4 = client.patch(f"/api/trips/{trip_id}/status", json={"status": "inside_plant"})
        self.assertEqual(r4.status_code, 200)
        self.assertEqual(r4.json()["status"], "inside_plant")

    def test_06_vehicles_and_drivers(self):
        # Create vehicle
        veh_payload = {
            "plate": "KA01AB9999",
            "type": "Truck",
            "owner": "Logistics Hub",
            "transporter": "Apex Haulage",
            "authorized": True
        }
        r1 = client.post("/api/vehicles", json=veh_payload)
        self.assertEqual(r1.status_code, 201)
        v_id = r1.json()["id"]

        # Update vehicle
        r2 = client.patch(f"/api/vehicles/{v_id}", json={"status": "In Maintenance"})
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.json()["status"], "In Maintenance")

        # Delete vehicle
        r3 = client.delete(f"/api/vehicles/{v_id}")
        self.assertEqual(r3.status_code, 204)

        # Create driver
        d_payload = {
            "name": "Karthik R",
            "license": "DL-12-2022-0091",
            "phone": "+91 91234 56789",
            "vehicle": "KA05MZ5678"
        }
        r4 = client.post("/api/drivers", json=d_payload)
        self.assertEqual(r4.status_code, 201)

    def test_07_alerts_and_review(self):
        # Alerts
        r_alerts = client.get("/api/alerts")
        self.assertEqual(r_alerts.status_code, 200)
        alerts_list = r_alerts.json()
        if alerts_list:
            a_id = alerts_list[0]["id"]
            r_ack = client.patch(f"/api/alerts/{a_id}/read")
            self.assertEqual(r_ack.status_code, 200)
            self.assertTrue(r_ack.json()["isRead"])

        # Review Queue
        r_rev = client.get("/api/review")
        self.assertEqual(r_rev.status_code, 200)
        rev_list = r_rev.json()
        if rev_list:
            r_id = rev_list[0]["id"]
            r_cor = client.post(f"/api/review/{r_id}/correct", json={"correctedPlate": "WB12AB1234"})
            self.assertEqual(r_cor.status_code, 200)
            self.assertEqual(r_cor.json()["plate"], "WB12AB1234")

    def test_08_cameras_and_reports(self):
        # Cameras
        r_cam = client.get("/api/cameras")
        self.assertEqual(r_cam.status_code, 200)

        r_new_cam = client.post("/api/cameras", json={"name": "North Gate Lane 1", "gate": "Gate 01", "direction": "entry"})
        self.assertEqual(r_new_cam.status_code, 201)

        # Reports overview
        r_rep = client.get("/api/reports/overview")
        self.assertEqual(r_rep.status_code, 200)
        rep_data = r_rep.json()
        for key in ["gateVolume", "transporterVolume", "dwellTrend", "decisions", "repeatVisitors", "overstays", "correctedReads", "totalReads"]:
            self.assertIn(key, rep_data)


if __name__ == "__main__":
    unittest.main()
