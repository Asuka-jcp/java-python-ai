# Java + FastAPI AI Demo

## 架构
- 前端: HTML + JS + CSS
- Java Spring Boot: 参数校验、聚合接口、调用 Python 服务
- Python FastAPI: 专项能力（热点和改写）

## 启动
### 1) 启动 Python
```bash
cd python-service
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2) 启动 Java
```bash
mvn spring-boot:run
```

打开 `http://localhost:8080`.

## 可选AI配置
若要让 Python 真正调用大模型改写，设置:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (可选)
- `OPENAI_MODEL` (可选)
