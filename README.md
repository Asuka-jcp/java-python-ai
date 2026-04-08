# Java + FastAPI AI Demo

## 架构
- 前端: HTML + JS + CSS
- Java Spring Boot: 参数校验、聚合接口、调用 Python 服务
- Python FastAPI: 专项能力（热点、改写、对话意图路由）

## 启动
### 1) 启动 Python
```bash
cd python-service
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python start.py
```

### 2) 启动 Java
```bash
mvn spring-boot:run
```

打开 `http://localhost:8080`。

## 可选 AI 配置
若要让 Python 使用大模型进行意图识别与改写，设置：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (可选)
- `OPENAI_MODEL` (可选)

## API
- Java 网关：
  - `GET /api/gateway/platforms`
  - `GET /api/gateway/hot-topics?platform=...`
  - `POST /api/gateway/rewrite`
  - `POST /api/gateway/chat`
- Python 服务：
  - `POST /api/ai/hot-topics`
  - `POST /api/ai/rewrite`
  - `POST /api/ai/chat`

## 新增：AI 对话能力
- 前端新增聊天窗口，用户可输入自然语言。
- Python 会让 LLM 判断意图（热点查询 / 文章改写 / 普通问答），并自动调用已有接口。
- 返回自然语言结果给前端。
