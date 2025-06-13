# Remote Eval Demo

一个最小可运行的 "Laptop 作为服务器、内网机器轮询" 通信示例。

## 目录

```
examples/communication_demo/
 ├─ server_demo.py  # FastAPI 服务端
 ├─ worker_demo.py  # 内网 worker 轮询脚本
 └─ README.md       # 当前文档
```

## 运行前准备

```bash
# 任意终端（两端都要）
pip install fastapi uvicorn requests
```

环境变量（两端一致）：
| 变量             | 说明                | 默认 |
|------------------|---------------------|------|
| `COMM_DEMO_TOKEN`| Bearer 鉴权 Token   | my-secret-token |

额外，worker 端可设：
| 变量             | 说明                  | 默认 |
|------------------|-----------------------|------|
| `COMM_DEMO_BASE` | 服务器地址            | http://127.0.0.1:8000 |
| `COMM_DEMO_POLL` | 轮询间隔（秒）        | 2 |

## 步骤 1：在笔记本启动服务端

```bash
export COMM_DEMO_TOKEN="my-secret-token"
python examples/communication_demo/server_demo.py
```

若需要公网可达，可用 Cloudflare Tunnel / ngrok 等，将本地 `8000` 端口暴露并记住外部 URL，例如 `https://abc.tunnel.dev`。

## 步骤 2：在内网服务器启动 worker

```bash
# BASE_URL 改成上一步拿到的公网 https 地址
export COMM_DEMO_BASE="https://abc.tunnel.dev"
export COMM_DEMO_TOKEN="my-secret-token"
python examples/communication_demo/worker_demo.py
```

## 步骤 3：提交任务并查看结果

在服务端再开一个终端（或用 Postman/curl）：

```bash
curl -X POST http://127.0.0.1:8000/enqueue \
     -H "Authorization: my-secret-token" \
     -H "Content-Type: application/json" \
     -d '{"job_id":"42","payload":"hello world"}'
```

服务端日志会看到排队信息；worker 端会在下一轮轮询时收到任务，将 payload 转成大写 "HELLO WORLD" 后回传。

查询结果：

```bash
curl http://127.0.0.1:8000/result/42 -H "Authorization: my-secret-token"
```

输出：
```json
{"job_id":"42","output":"HELLO WORLD"}
```

至此通信链路打通，后续把 `payload` 换成补丁/文件路径、把 worker 的处理逻辑替换为 `evaluator.py` 即可。 