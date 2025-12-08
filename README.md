# 🚅 RAG Line Bot - 淡海輕軌維修支援助手

這是一個結合 **RAG (Retrieval-Augmented Generation)** 技術與 **Line Messaging API** 的智慧問答機器人專案。
專案使用 **Qwen 2.5 (1.5B)** 作為生成模型，旨在協助處理淡海輕軌的維修與故障排除查詢（如牽引動力故障排除流程）。

## 🛠️ 技術架構 (Tech Stack)

- **語言 (Language):** Python 3.13+
- **框架 (Framework):** FastAPI / Uvicorn (非同步處理)
- **LLM 模型:** Qwen/Qwen2.5-1.5B-Instruct (支援 Mac MPS 加速)
- **RAG 機制:** - 知識庫格式: JSONL (`knowledge.jsonl`)
  - 檢索方法: 向量相似度搜尋 (Vector Embeddings)

## 🚀 快速開始 (Quick Start)

### 1. 環境設定
建議使用虛擬環境 (Virtual Environment) 執行此專案。

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate

# 安裝依賴套件
pip install -r requirements.txt
