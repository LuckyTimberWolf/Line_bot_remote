import os
import sys
import torch
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 載入環境變數
load_dotenv()

# 2. 取得密鑰與設定
channel_secret = os.getenv('LINE_CHANNEL_SECRET')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
hf_token = os.getenv('HF_TOKEN') # Hugging Face Token (Gemma 需要申請權限)

if not all([channel_secret, channel_access_token, hf_token]):
    print("錯誤：請確認 .env 檔案中已設定 LINE 密鑰與 HF_TOKEN")
    sys.exit(1)

# 3. 初始化 LINE Bot API
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# ==========================================
# [AI 教授重點教學] 初始化 Hugging Face 模型
# 我們將模型載入放在 Global Scope，這樣 Cloud Run 啟動時只會載入一次
# ==========================================
MODEL_ID = "google/gemma-3-270m-it"  # 若有 270m 版本請在此更換 ID

print(f"正在載入模型 {MODEL_ID} ... (這可能需要幾分鐘)")
try:
    # 設定 device: 如果有 GPU 用 cuda，Mac 用 mps，雲端 Cloud Run 通常是 cpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32, # CPU通常用 float32 較穩
        low_cpu_mem_usage=True
    ).to(device)
    print("模型載入完成！")
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
        # --- LLM 生成邏輯 ---

        # 1. 建立 Prompt (Gemma 建議的對話格式)

        '''
        # 教授修改：在使用者訊息前，加入 "請用繁體中文回答" 的強硬指令
        system_instruction = "請扮演一位繁體中文的 AI 助理，無論使用者用什麼語言，你一律都要使用「繁體中文 (Traditional Chinese)」回答，並且不要使用簡體字。"
        
        chat = [
            { "role": "user", "content": system_instruction + "\n使用者說：" + user_msg },
        ]
        '''

        # 原始沒有加入系統指令的版本
        # 1. 建立 Prompt (Gemma 建議的對話格式)
        chat = [
            { "role": "user", "content": user_msg },
        ]
        

        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        # 2. 轉為 Tensor
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        # 3. 生成回應
        # max_new_tokens 控制生成長度，避免超時
        outputs = model.generate(input_ids=inputs, max_new_tokens=150)

        # 4. 解碼
        # 這裡需要去除掉 input 的部分，只留新生成的文字
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 簡單處理，移除 prompt 部分（依模型不同可能需要調整字串處理）
        response_text = generated_text.replace(prompt, "").strip()
        
        # 如果 response_text 包含原始 prompt，可以做更細緻的切分，這裡簡化處理：
        # Gemma 的 chat template 有時會直接輸出內容，我們取最後一段
        final_reply = generated_text.split("model\n")[-1] 
        final_reply = final_reply.replace("**", "")  # 把粗體符號拿掉

    except Exception as e:
        print(f"生成錯誤: {e}")
        final_reply = "抱歉，我現在有點暈，無法思考。"

    # 回覆訊息
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=final_reply)
    )