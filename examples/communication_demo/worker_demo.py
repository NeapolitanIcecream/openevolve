import os
import time
import uuid
from typing import Optional

import requests

BASE_URL = os.environ.get("COMM_DEMO_BASE", "http://127.0.0.1:8000")
API_TOKEN = os.environ.get("COMM_DEMO_TOKEN", "my-secret-token")
POLL_INTERVAL = float(os.environ.get("COMM_DEMO_POLL", 2))  # 秒

HEADERS = {"Authorization": API_TOKEN}

def log(msg: str):
    print(f"[worker] {msg}")


def main():
    while True:
        try:
            r = requests.get(f"{BASE_URL}/next", headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"fetch next task error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if data.get("status") == "empty":
            time.sleep(POLL_INTERVAL)
            continue

        job_id = data["job_id"]
        payload = data["payload"]
        log(f"got task {job_id}: payload='{payload}'")

        # 这里执行真正的 evaluator；demo 里直接 upper-case
        output = payload.upper()
        time.sleep(1)  # 模拟耗时

        try:
            res = requests.post(
                f"{BASE_URL}/result",
                headers=HEADERS,
                json={"job_id": job_id, "output": output},
                timeout=10,
            )
            res.raise_for_status()
            log(f"sent result for {job_id}: {output}")
        except Exception as e:
            log(f"send result error: {e}")


if __name__ == "__main__":
    main() 