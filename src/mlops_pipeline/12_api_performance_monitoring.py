
import json
import time
import psutil
from pathlib import Path
from datetime import datetime

import httpx


BASE_DIR = Path(__file__).resolve().parents[2]

API_URL = "http://127.0.0.1:8000"

REPORT_DIR = BASE_DIR / "artifacts"
REPORT_DIR.mkdir(exist_ok=True)

response_times = []
successful_requests = 0
failed_requests = 0

test_data = {
    "Company": "A. Morin",
    "Specific_Bean_Origin": "Peru",
    "Review_Date": 2016,
    "Cocoa_Percent": 70,
    "Company_Location": "France",
    "Bean_Type": "",
    "Broad_Bean_Origin": ""
}

print("Starting API Performance Monitoring")

with httpx.Client(timeout=30) as client:

    for i in range(10):

        start_time = time.time()

        try:
            response = client.post(
                API_URL + "/predict",
                json=test_data
            )

            end_time = time.time()

            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)

            if response.status_code == 200:
                successful_requests += 1
            else:
                failed_requests += 1

            print(
                f"Request {i+1}: "
                f"Status={response.status_code}, "
                f"Time={response_time:.2f} ms"
            )

        except Exception as error:
            failed_requests += 1
            print(f"Request {i+1} failed: {error}")


average_time = (
    sum(response_times) / len(response_times)
    if response_times else 0
)

report = {
    "timestamp": datetime.now().isoformat(),
    "total_requests": 10,
    "successful_requests": successful_requests,
    "failed_requests": failed_requests,
    "average_response_time_ms": round(average_time, 2),
    "cpu_usage_percent": psutil.cpu_percent(interval=1),
    "memory_usage_percent": psutil.virtual_memory().percent
}

report_path = REPORT_DIR / "api_performance_report.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\nPERFORMANCE REPORT")
print(json.dumps(report, indent=4))

print("\nReport saved:", report_path)
