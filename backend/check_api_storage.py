"""
REST API Database Storage Check for ANPRX

Tests data flow over the HTTP REST API on http://localhost:5001:
1. Health endpoint & DB connectivity.
2. Querying live operations (Vehicles, Trips, Entry/Exit logs).
3. Querying Dashboard summary & real-time activity feed from MySQL.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

# Configure UTF-8 safe stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://localhost:5001"


def make_request(path: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(data).encode("utf-8") if data else None
    
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            return {"status": resp.getcode(), "data": json.loads(content) if content else {}}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return {"status": e.code, "error": body}
    except Exception as e:
        return {"status": 0, "error": str(e)}


def test_api_storage():
    print("=" * 70)
    print("      ANPRX REST API & DATABASE STORAGE VERIFICATION")
    print("=" * 70)
    
    # 1. Health check
    print("\n[1] Checking /api/healthz...")
    res = make_request("/api/healthz")
    if res["status"] != 200:
        print(f"[-] Backend not responding on {BASE_URL}. Is backend running?")
        print(f"    Error: {res.get('error')}")
        return False
    print(f"[+] Backend Healthy: {res['data']}")

    # 2. Get Vehicles & Master Data
    print("\n[2] Querying Vehicles via /api/vehicles...")
    res = make_request("/api/vehicles")
    if res["status"] == 200:
        vehicles = res["data"]
        print(f"[+] Retrieved {len(vehicles)} vehicles from database via API.")
        if vehicles:
            sample = vehicles[0]
            print(f"    Sample: ID={sample.get('id')}, Plate={sample.get('plate')}, Type={sample.get('type')}, Owner={sample.get('owner')}")
    else:
        print(f"[-] /api/vehicles returned: {res}")

    # 3. Get Scheduled Trips
    print("\n[3] Querying Scheduled Trips via /api/trips...")
    res = make_request("/api/trips")
    if res["status"] == 200:
        trips = res["data"]
        print(f"[+] Retrieved {len(trips)} scheduled trips from database via API.")
        if trips:
            sample = trips[0]
            print(f"    Sample: Trip#{sample.get('tripNumber', sample.get('id'))}, Plate={sample.get('plate')}, Driver={sample.get('driver')}, Status={sample.get('status')}")
    else:
        print(f"[-] /api/trips returned: {res}")

    # 4. Get Entry/Exit Logs (Events)
    print("\n[4] Querying Entry/Exit Events via /api/events...")
    res = make_request("/api/events")
    if res["status"] == 200:
        events = res["data"]
        print(f"[+] Retrieved {len(events)} stored entry/exit events from database via API.")
        for ev in events[:3]:
            print(f"    * Event #{ev.get('id')}: Plate={ev.get('plate')}, Decision={ev.get('decision')}, Status={ev.get('status')}")
    else:
        print(f"[-] /api/events returned: {res}")

    # 5. Get Live Dashboard Summary Metrics
    print("\n[5] Querying Dashboard Summary via /api/dashboard/summary...")
    res = make_request("/api/dashboard/summary")
    if res["status"] == 200:
        summary = res["data"]
        print("[+] Dashboard Summary retrieved from MySQL:")
        print(f"    * Vehicles Inside:       {summary.get('vehiclesInside')}")
        print(f"    * Entries Today:         {summary.get('entriesToday')}")
        print(f"    * Exits Today:           {summary.get('exitsToday')}")
        print(f"    * Active Alerts:         {summary.get('activeAlerts')}")
        print(f"    * Recognition Accuracy:  {summary.get('recognitionAccuracy', 0) * 100:.1f}%")
        print(f"    * Avg Dwell Time:        {summary.get('avgDwellMinutes')} mins")
        print(f"    * Gates Online:          {summary.get('gatesOnline')}/{summary.get('totalGates')}")
    else:
        print(f"[-] /api/dashboard/summary returned: {res}")

    # 6. Get Activity Feed
    print("\n[6] Querying Activity Feed via /api/dashboard/activity...")
    res = make_request("/api/dashboard/activity")
    if res["status"] == 200:
        activity = res["data"]
        print(f"[+] Retrieved {len(activity)} real-time feed items from database via API.")
        for act in activity[:3]:
            print(f"    * {act.get('title')} ({act.get('detail')})")
    else:
        print(f"[-] /api/dashboard/activity returned: {res}")

    print("\n" + "=" * 70)
    print("  >>> REST API & DATABASE STORAGE VERIFICATION COMPLETE! <<<")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_api_storage()
    sys.exit(0 if success else 1)
