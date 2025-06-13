from collections import deque
import os
import uuid
from typing import Deque, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

# -------------------------------------------------
# 简易配置（可通过环境变量覆盖）
# -------------------------------------------------
API_TOKEN = os.environ.get("COMM_DEMO_TOKEN", "knowledge42")

app = FastAPI(title="Remote Eval Demo – Server", version="0.1")

# 任务队列与结果缓存（内存）
_task_queue: Deque[Dict] = deque()
_results: Dict[str, Dict] = {}


# ---------------- Pydantic 模型 ------------------
class Task(BaseModel):
    job_id: str
    payload: str  # 这里用简单字符串，实际可放补丁 diff 等


class Result(BaseModel):
    job_id: str
    output: str


# ------------------ 工具函数 ---------------------

def _check_auth(auth_header: Optional[str]):
    """简单 Bearer Token 校验"""
    if auth_header != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ------------------ 路由定义 --------------------


@app.post("/enqueue", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_task(task: Task, authorization: str = Header(None)):
    """外部接口：添加新任务到队列"""
    _check_auth(authorization)
    _task_queue.append(task.dict())
    return {"status": "queued", "queue_length": len(_task_queue)}


@app.get("/next")
async def get_next_task(authorization: str = Header(None)):
    """被 worker 轮询：获取下一任务"""
    _check_auth(authorization)
    if not _task_queue:
        return {"status": "empty"}
    task = _task_queue.popleft()
    return task


@app.post("/result")
async def post_result(res: Result, authorization: str = Header(None)):
    """worker 上传结果"""
    _check_auth(authorization)
    _results[res.job_id] = res.dict()
    return {"status": "received"}


@app.get("/result/{job_id}")
async def fetch_result(job_id: str, authorization: str = Header(None)):
    """查询结果（可选）"""
    _check_auth(authorization)
    if job_id not in _results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not found")
    return _results[job_id]


# ----------------- CLI 运行提示 ------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("COMM_DEMO_PORT", 8042))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
