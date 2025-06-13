import os
import time
import uuid
from typing import Optional

import requests
import urllib3

BASE_URL = os.environ.get("COMM_DEMO_BASE", "https://evolve.voile.tech")
API_TOKEN = os.environ.get("COMM_DEMO_TOKEN", "knowledge42")
POLL_INTERVAL = float(os.environ.get("COMM_DEMO_POLL", 2))  # 秒

# ------- TLS / CA 设置 -------
# 如果处于自签名 CA 的内网 proxy 后面，可以通过环境变量 COMM_DEMO_CA_BUNDLE 指定 CA 证书路径；
# 如果未设置，则默认关闭证书校验（不推荐，仅用于测试）。
CA_BUNDLE = os.environ.get("COMM_DEMO_CA_BUNDLE")
# verify=True | False | "/path/to/ca.pem"
if CA_BUNDLE:
    VERIFY = CA_BUNDLE  # 指定 CA bundle 路径
else:
    VERIFY = False      # 关闭校验
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {"Authorization": API_TOKEN}

def log(msg: str):
    print(f"[worker] {msg}")


def main():
    while True:
        try:
            r = requests.get(
                f"{BASE_URL}/next", headers=HEADERS, timeout=10, verify=VERIFY
            )
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
                verify=VERIFY,
            )
            res.raise_for_status()
            log(f"sent result for {job_id}: {output}")
        except Exception as e:
            log(f"send result error: {e}")


if __name__ == "__main__":
    main() 