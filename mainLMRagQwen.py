import os
import sys
import json  # 必須匯入，用於解析 JSONL
import torch
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from opencc import OpenCC  # 繁體轉換
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer # <--- 新增 TextStreamer

# --- RAG 相關套件 ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # 這是新版的正確位置

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
# [AI 教授重點教學] RAG 知識庫初始化
# ==========================================
print("正在建立 RAG 知識庫 (FAISS)...")

try:
    # A. 讀取 knowledge.jsonl 並進行結構化處理
    documents = []
    
    if not os.path.exists("knowledge.jsonl"):
        print("警告：找不到 knowledge.jsonl，將使用空資料。")
    else:
        with open("knowledge.jsonl", "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue  # 跳過空行
                
                try:
                    # 嘗試解析 JSON
                    data = json.loads(line)
                    
                    # [教授修正 1]：相容多種欄位名稱 (symptom/solution 或 instruction/response)
                    instruction = data.get('symptom', data.get('instruction', ''))
                    response = data.get('solution', data.get('response', ''))
                    
                    # 組合出乾淨的文字格式，去除 JSON 符號干擾
                    page_content = f"故障症狀：{instruction}\n排除方法：{response}"
                    
                    # 建立 LangChain 的 Document 物件
                    doc = Document(
                        page_content=page_content,
                        metadata={"source": "knowledge.jsonl", "row": line_number}
                    )
                    documents.append(doc)
                except json.JSONDecodeError:
                    print(f"警告：第 {line_number} 行格式錯誤，已跳過。")

    # B. 設定文件列表 (不需要切分 splitter，因為每一行已經是獨立知識點)
    docs = documents
    print(f"成功載入 {len(docs)} 條知識片段！")
    
    # 檢查第一條資料確認格式正確 (除錯用)
    if docs:
        print(f"範例資料片段: {docs[0].page_content[:50]}...")

    # C. 載入 Embedding 模型
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # [教授修正] 改用多語言模型，這樣才分得清「牽引」跟「集電弓」的差別
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # D. 建立向量資料庫
    if docs:
        vector_db = FAISS.from_documents(docs, embeddings)
        print("知識庫建立完成！")
    else:
        print("警告：沒有載入任何文件，知識庫為空。")
        vector_db = None

except Exception as e:
    print(f"RAG 初始化失敗: {e}")
    vector_db = None

# ==========================================
# 初始化 LLM 模型 (Gemma)
# ==========================================
#MODEL_ID = "google/gemma-3-270m-it"

# [教授推薦] 改用 Qwen 2.5 (1.5B)，中文能力與邏輯大幅提升
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"正在載入生成模型 {MODEL_ID} ...")
try:
    # 1. 優先嘗試使用 MPS (Mac GPU)
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 啟動 Mac GPU 硬體加速 (MPS)")
    else:
        device = "cpu"
        print("🐢 使用 CPU 模式")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    
    # 2. [關鍵修正] 強制使用 float32 載入模型
    # 這能解決 Mac MPS 出現 "probability tensor contains nan" 的錯誤
    # 雖然比 float16 佔記憶體，但比 CPU 快非常多且穩定
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
        # --- [RAG 檢索階段] ---
        rag_context = ""
        if vector_db:
            # 搜尋最相關的 2 段文字
            search_results = vector_db.similarity_search(user_msg, k=3)
            if search_results:
                rag_context = "\n".join([res.page_content for res in search_results])
                print(f"搜尋到的相關知識: {rag_context[:100]}...") 

        # --- [Prompt 組合階段] ---
        if rag_context:
            full_prompt_msg = (
                f"你是一位捷運維修專家。請根據以下【維修手冊】回答問題。\n"
                f"【維修手冊】：\n{rag_context}\n\n"
                f"問題：{user_msg}\n"
                f"回答："
            )
        else:
            full_prompt_msg = (
                f"你是一位助理。請回答問題：{user_msg}"
            )

        # --- [LLM 生成階段] ---
        chat = [
            { "role": "user", "content": full_prompt_msg },
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        # [教授修正 3]：使用精準切割法 (Input Length Slicing)
        input_length = inputs.input_ids.shape[1]
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=400,
            repetition_penalty=1.1,  # 防止重複
            do_sample=True,          # 讓回答稍微自然一點
            temperature=0.3,          # 降低隨機性，專注於手冊
            streamer=streamer        # <--- 加入這一行
        )

        # 只解碼「新生成的」token，徹底解決 split 切割錯誤問題
        generated_tokens = outputs[0][input_length:]
        final_reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # --- [後處理階段] ---
        cc = OpenCC('s2t')
        final_reply = cc.convert(final_reply)
        final_reply = final_reply.replace("**", "").strip()

        if not final_reply or final_reply.strip() == "":
            final_reply = "抱歉，我正在思考中，但暫時無法產生回應。請再試一次或提供更多資訊。"

    except Exception as e:
        print(f"生成錯誤: {e}")
        final_reply = "抱歉，系統發生錯誤。"

    # 回覆訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )