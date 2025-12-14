import os
import sys
import json
import jieba  # 新增：用於中文斷詞
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from openai import OpenAI

# --- RAG 與 檢索 相關套件 ---
from langchain_huggingface import HuggingFaceEmbeddings # 更新引入路徑
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# 1. 載入環境變數
load_dotenv()

channel_secret = os.getenv('LINE_CHANNEL_SECRET')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
openai_api_key = os.getenv('OPENAI_API_KEY')

# 設定模型名稱 (目前 OpenAI 最新為 gpt-4o，若未來 gpt-5.2 發布可在此修改)
OPENAI_MODEL = "gpt-4o" 

if not all([channel_secret, channel_access_token, openai_api_key]):
    print("錯誤：請確認 .env 檔案中已設定 LINE 密鑰與 OPENAI_API_KEY")
    sys.exit(1)

# 2. 初始化 LINE Bot 與 OpenAI Client
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)
client = OpenAI(api_key=openai_api_key)

# ==========================================
# [AI 教授重點教學] RAG 混合檢索系統初始化
# ==========================================
print("正在建立 RAG 混合檢索系統 (FAISS + BM25)...")

ensemble_retriever = None

# 定義中文斷詞函數 (給 BM25 使用，解決中文黏在一起導致檢索不到的問題)
def chinese_tokenizer(text):
    return list(jieba.cut(text))

try:
    documents = []
    knowledge_file = "knowledgeNEW.jsonl"
    
    if not os.path.exists(knowledge_file):
        print(f"警告：找不到 {knowledge_file}，將使用空資料。")
    else:
        with open(knowledge_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    # 容錯處理：支援 symptom/instruction 兩種欄位命名
                    instruction = data.get('symptom', data.get('instruction', ''))
                    response = data.get('solution', data.get('response', ''))
                    
                    # 組合內容供檢索
                    page_content = f"故障症狀：{instruction}\n排除方法：{response}"
                    
                    doc = Document(
                        page_content=page_content,
                        metadata={"source": knowledge_file, "row": line_number}
                    )
                    documents.append(doc)
                except json.JSONDecodeError:
                    print(f"警告：第 {line_number} 行格式錯誤，已跳過。")

    docs = documents
    print(f"成功載入 {len(docs)} 條知識片段！")

    if docs:
        # A. 語意檢索 (Vector Search) - 使用本地 BGE-M3 模型
        print("正在載入 Embedding 模型 (BAAI/bge-m3)...")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        
        vector_db = FAISS.from_documents(docs, embeddings)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # B. 關鍵字檢索 (Keyword Search) - BM25 + Jieba 斷詞
        print("正在建立 BM25 索引 (含 Jieba 斷詞)...")
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=chinese_tokenizer  # 關鍵修正：加入中文斷詞
        )
        bm25_retriever.k = 3

        # C. 混合檢索 (Ensemble)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # 權重可依實際測試調整
        )
        print("混合檢索系統 (Hybrid Search) 建立完成！")
    else:
        print("警告：沒有載入任何文件，知識庫為空。")

except Exception as e:
    print(f"RAG 初始化失敗: {e}")
    # 不強制退出，讓 Server 仍能啟動，但檢索功能會失效
    pass

app = FastAPI()

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_msg = event.message.text.strip()
    print(f"收到使用者訊息: {user_msg}")

    try:
        # --- [1. RAG 混合檢索階段] ---
        rag_context = ""
        if ensemble_retriever:
            # 執行混合檢索
            found_docs = ensemble_retriever.invoke(user_msg)
            if found_docs:
                # 取前 2 筆最相關資料
                top_docs = found_docs[:2]
                rag_context = "\n\n".join([f"【故障排除手冊參考資料】:\n{doc.page_content}" for i, doc in enumerate(top_docs)])
                # 
                print(f"--- 🔍 檢索到的參考資料 ---\n{rag_context[:100]}...\n-----------------------------")
            else:
                print("--- 🔍 未檢索到相關資料 ---")

        # --- [2. Prompt 組合階段] ---
        system_prompt = (
            "你是一位資深的捷運維修專家，負責協助維修人員排除故障。"
            "請嚴格根據提供給你的【故障排除手冊】來回答使用者的問題。"
            "回答規則："
            "1. 若參考資料中有對應的故障代碼或症狀，請列出具體的排除步驟。"
            "2. 若參考資料與問題無關或不足以回答，請明確回答「手冊中查無此故障資料，建議查閱實體手冊或聯繫行控中心」，嚴禁自行編造。"
            "3. 語氣請保持專業、冷靜，並使用繁體中文。"
        )

        if rag_context:
            user_content = f"參考資料：\n{rag_context}\n\n使用者問題：{user_msg}"
        else:
            user_content = f"使用者問題：{user_msg} (注意：系統未檢索到相關手冊資料)"

        # --- [3. ChatGPT API 呼叫階段] ---
        print(f"正在呼叫 OpenAI API (Model: {OPENAI_MODEL})...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # 使用變數設定的模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1, # 降低溫度以確保回答基於事實
            max_tokens=600
        )

        final_reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"系統錯誤: {e}")
        final_reply = "抱歉，目前維修AI系統遭遇內部錯誤，請通知管理員。"

    # 回覆 LINE 訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )
    print("已回覆使用者訊息。")