import requests
import json
import time

url = "http://localhost:5001/api/video/process"
video_file = "static/processed_videos/processed_1787158922_upload_1787158921_numberplate1.jpg 2026-08-18 14-42-27.mp4"

print("Submitting video to live running backend at http://localhost:5001/api/video/process...")
t0 = time.time()
with open(video_file, "rb") as f:
    files = {"video": ("test_user_car.mp4", f, "video/mp4")}
    data = {"conf_threshold": "0.25", "max_process_fps": "15"}
    res = requests.post(url, files=files, data=data)

t1 = time.time()
print(f"Status Code: {res.status_code} (took {t1 - t0:.2f}s)")
if res.status_code == 200:
    resp = res.json()
    print("Success:", resp.get("success"))
    print("Vehicles Detected:", resp.get("vehicles_detected"))
    print("Plates Detected:", resp.get("plates_detected"))
    detections = resp.get("detections", [])
    for d in detections:
        print(f"  • Track #{d['track_id']}: {d['vehicle_class']} (Conf: {d['vehicle_confidence']*100:.1f}%) | Plate: {d['plate_number']} (OCR Conf: {d['ocr_confidence']*100:.1f}%)")
else:
    print("Error:", res.text)
