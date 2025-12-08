import os
import sys
import torch
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from opencc import OpenCC  # 繁體轉換

# --- RAG 相關套件 ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

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
    # A. 讀取 knowledge.txt
    if not os.path.exists("knowledge.txt"):
        # 建立一個預設檔案避免報錯
        with open("knowledge.txt", "w", encoding="utf-8") as f:
            f.write("這是預設的知識庫內容。目前沒有特定資料。")
    
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    documents = loader.load()

    # B. 切分文字 (避免文章太長，模型吃不下)

    text_splitter = CharacterTextSplitter(
            separator="},",   # 依照 JSON 物件的逗號切分
            chunk_size=300, # 每段 300 字元
            chunk_overlap=0 # 不重疊
    )
    #text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # C. 載入 Embedding 模型 (負責把文字變成向量)
    # 使用輕量級的 sentence-transformers，適合本機運行
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # D. 建立向量資料庫
    vector_db = FAISS.from_documents(docs, embeddings)
    print("知識庫建立完成！")

except Exception as e:
    print(f"RAG 初始化失敗: {e}")
    # 若失敗，為了不讓程式掛掉，設為 None
    vector_db = None

# ==========================================
# 初始化 LLM 模型 (Gemma)
# ==========================================
MODEL_ID = "google/gemma-3-270m-it"

print(f"正在載入生成模型 {MODEL_ID} ...")
try:
    device = "cpu" # 預設為 CPU
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): # 支援 Mac M1/M2/M3
        device = "mps"
    '''
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.float32,  # <--- 關鍵修改：強制全精度，穩如泰山
        #torch_dtype=torch.float16 if device != "cpu" else torch.float32,
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
            # 搜尋最相關的 2 段文字 (增加 k 可以讓它讀多一點，但 270M 記憶體有限，先維持 2)
            search_results = vector_db.similarity_search(user_msg, k=2)
            if search_results:
                rag_context = "\n".join([res.page_content for res in search_results])
                # 為了除錯，我們把它印出來看看到底抓到了什麼
                print(f"搜尋到的相關知識: {rag_context[:100]}...") 

        # --- [Prompt 組合階段] ---
        # 針對 270M 小模型的優化指令
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

        # --- [LLM 生成階段] (修復 Attention Mask 警告) ---
        chat = [
            { "role": "user", "content": full_prompt_msg },
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # ✅ 修改點：改用 tokenizer 直接回傳 tensor，並取得 attention_mask
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # <--- 加入這行消除警告
            max_new_tokens=400  # 長度
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_reply = generated_text.split("model\n")[-1]

        # --- [後處理階段] 簡轉繁 ---
        cc = OpenCC('s2t')
        final_reply = cc.convert(final_reply)
        final_reply = final_reply.replace("**", "")

        # 🛑 [教授新增] 防呆機制：如果模型回答空白，手動塞一句話
        if not final_reply or final_reply.strip() == "":
            print("警告：模型生成了空字串，使用預設回覆。")
            final_reply = "抱歉，我正在思考中，但暫時無法產生回應。請再試一次或提供更多資訊。"

    except Exception as e:
        print(f"生成錯誤: {e}")
        final_reply = "抱歉，系統發生錯誤。"

    # 回覆訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )