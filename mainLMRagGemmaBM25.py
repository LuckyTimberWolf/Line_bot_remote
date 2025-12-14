import os
import sys
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from opencc import OpenCC  # 繁體轉換

# --- RAG 與 檢索 相關套件 ---
# --- 標準化 Import (請直接覆蓋舊的 Import 區塊) ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
# 在 LangChain 0.3.0+，EnsembleRetriever 位於標準路徑
from langchain.retrievers import EnsembleRetriever

# 1. 載入環境變數
load_dotenv()

# 2. 取得密鑰與設定
channel_secret = os.getenv('LINE_CHANNEL_SECRET') 
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
hf_token = os.getenv('HF_TOKEN')

if not all([channel_secret, channel_access_token, hf_token]):
    print("錯誤：請確認 .env 檔案中已設定 LINE 密鑰與 HF_TOKEN")
    sys.exit(1)

# 3. 初始化 LINE Bot API
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# ==========================================
# [AI 教授重點教學] RAG 混合檢索系統初始化
# ==========================================
print("正在建立 RAG 混合檢索系統 (FAISS + BM25)...")

ensemble_retriever = None  # 全域變數

try:
    # A. 讀取 knowledge.jsonl 並進行結構化處理
    documents = []
    
    if not os.path.exists("knowledge.jsonl"):
        print("警告：找不到 knowledge.jsonl，將使用空資料。")
    else:
        with open("knowledge.jsonl", "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    # 組合出乾淨的文字格式
                    instruction = data.get('symptom', data.get('instruction', ''))
                    response = data.get('solution', data.get('response', ''))
                    page_content = f"故障症狀：{instruction}\n排除方法：{response}"
                    
                    doc = Document(
                        page_content=page_content,
                        metadata={"source": "knowledge.jsonl", "row": line_number}
                    )
                    documents.append(doc)
                except json.JSONDecodeError:
                    print(f"警告：第 {line_number} 行格式錯誤，已跳過。")

    docs = documents
    print(f"成功載入 {len(docs)} 條知識片段！")

    if docs:
        # B. 載入 Embedding 模型 (用於語意理解)
        # 使用多語言模型以支援中文語意
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        # 修改後 (改用 BGE-M3，目前中文檢索的最強者之一)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # C. 建立兩種檢索器
        # 1. BM25 (關鍵字精準檢索) - 專治 "EB1", "總故障燈" 這種專有名詞
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3  # 取前 3 名

        # 2. FAISS (語意向量檢索) - 專治 "車子不動", "沒電" 這種模糊描述
        vector_db = FAISS.from_documents(docs, embeddings)
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        # D. 建立 Ensemble (混合) 檢索器
        # weights=[0.5, 0.5] 代表關鍵字和語意同樣重要
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        print("混合檢索系統 (Hybrid Search) 建立完成！")
    else:
        print("警告：沒有載入任何文件，知識庫為空。")

except Exception as e:
    print(f"RAG 初始化失敗: {e}")
    sys.exit(1)

# ==========================================
# 初始化 LLM 模型
# ==========================================
# 建議：若記憶體允許，將此處改為 "Qwen/Qwen2.5-1.5B-Instruct" 效果會更好
MODEL_ID = "google/gemma-3-270m-it" 

print(f"正在載入生成模型 {MODEL_ID} ...")
try:
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 啟動 Mac GPU 硬體加速 (MPS)")
    else:
        device = "cpu"
        print("🐢 使用 CPU 模式")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True
    ).to(device)
    
    print(f"生成模型載入完成！運行裝置: {device}")
except Exception as e:
    print(f"模型載入失敗: {e}")
    sys.exit(1)

app = FastAPI()

# 4. 設定 Webhook 入口
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers['X-Line-Signature']
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return 'OK'

# 5. 處理文字訊息的邏輯
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_msg = event.message.text
    print(f"收到訊息: {user_msg}")

    try:
        # --- [RAG 混合檢索階段] ---
        rag_context = ""
        found_docs = []

        if ensemble_retriever:
            # 使用 invoke 進行混合搜尋
            found_docs = ensemble_retriever.invoke(user_msg)
            
            if found_docs:
                # 為了避免 Context 太長，只取前 2 筆最相關的
                top_docs = found_docs[:2]
                rag_context = "\n\n".join([f"【參考資料 {i+1}】:\n{doc.page_content}" for i, doc in enumerate(top_docs)])
                
                # [除錯 Log] 印出找到什麼，確認 BM25 是否生效
                print(f"--- 🔍 檢索到的知識 (Top 2) ---")
                for doc in top_docs:
                    print(f"[內容]: {doc.page_content[:50]}...")
                print("-----------------------------")

        # --- [Prompt 組合階段] ---
        # 教授修正：使用更嚴謹的指令格式，防止模型瞎掰或接龍
        if rag_context:
            full_prompt_msg = (
                f"### 指令 ###\n"
                f"你是一位專業的捷運維修專家。請依據下方提供的【維修手冊片段】，回答使用者的問題。\n"
                f"規則 1：請直接列出排除步驟，不要廢話。\n"
                f"規則 2：若手冊內容與問題無關，請直接回答「查無相關維修資料」，不可自行編造。\n\n"
                f"### 維修手冊片段 ###\n{rag_context}\n\n"
                f"### 使用者問題 ###\n{user_msg}\n\n"
                f"### 你的專業回答 ###\n"
            )
        else:
            full_prompt_msg = f"你是一位助理。請回答問題：{user_msg}"

        # --- [LLM 生成階段] ---
        chat = [
            { "role": "user", "content": full_prompt_msg },
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        input_length = inputs.input_ids.shape[1]
        
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=300,      # 不需要太長，維修步驟通常很簡潔
            repetition_penalty=1.2,  # 提高懲罰，避免重複
            do_sample=True,
            temperature=0.1,         # 降低隨機性，讓回答更死板、精確
            streamer=streamer
        )

        generated_tokens = outputs[0][input_length:]
        final_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # --- [後處理階段] ---
        cc = OpenCC('s2t')
        final_reply = cc.convert(final_reply)
        final_reply = final_reply.replace("**", "").replace("###", "").strip()

        if not final_reply:
            final_reply = "抱歉，系統運算中，請稍後再試。"

    except Exception as e:
        print(f"生成錯誤: {e}")
        final_reply = "抱歉，系統發生錯誤。"

    # 回覆訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )

if __name__ == "__main__":
    import uvicorn
    # 啟動伺服器
    uvicorn.run(app, host="0.0.0.0", port=5000)