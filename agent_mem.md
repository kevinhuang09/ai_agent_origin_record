
**連接方式**：透過 **cloudflared** 建立加密通道，讓筆電端 agent 可以存取遠端 GPU 伺服器的 AI 運算服務。

---

## 🖥️ 節點分工

### 筆電端：`agent_main.py`
- **角色**：Agent 主控程式
- **狀態**：⏳ 尚未提供程式碼，待補充

### 遠端伺服器：`server.py`
- **角色**：Neuropeptide AI 運算與生物資訊分析服務
- **狀態**：✅ 已分析完成

---

## 🛠️ Server.py 技術棧

| 類別 | 技術/套件 |
|------|----------|
| Web 框架 | FastAPI + Uvicorn |
| 服務埠 | `0.0.0.0:8800` |
| 跨域設定 | CORSMiddleware（`allow_origins=["*"]`）|
| 生物資訊 | Biopython（ProtParam）、CD-HIT、CD-HIT-2D |
| 深度學習 | PyTorch、Ankh embedding、ESMFold API |
| 視覺化 | Matplotlib |
| LLM 整合 | Ollama Proxy（反向代理到 `127.0.0.1:11434`）|

---

## 🗂️ 關鍵路徑設定

```python
OLD_ENV_PYTHON   = "/home/g503/miniconda3/envs/neuropredictor-tool/bin/python"
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MAIN_PROGRAM_DIR = BASE_DIR
IMAGE_DIR        = os.path.join(MAIN_PROGRAM_DIR, "plots")

# 🧬 Neuropeptide Agent 架構文件

> 建立日期：2026-05-15  
> 作者：黃瑞麟  
> 版本：v2.0（含 agent_main.py 完整解析）

---

## 📐 整體架構總覽

