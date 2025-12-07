import os
import sys
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv

# 1. 載入環境變數
load_dotenv()

# 2. 取得密鑰
channel_secret = os.getenv('LINE_CHANNEL_SECRET')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')

if channel_secret is None or channel_access_token is None:
    print("錯誤：請確認 .env 檔案中已設定 SECRET 和 TOKEN")
    sys.exit(1)

# 3. 初始化 LINE Bot API
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

app = FastAPI()

# 4. 設定 Webhook 入口 (LINE 會把訊息丟到這裡)
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers['X-Line-Signature'] # 取得簽名
    body = await request.body() # 取得訊息內容
    
    try:
        # 驗證訊息是否真的來自 LINE (安全性檢查)
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    return 'OK'

# 5. 處理文字訊息的邏輯
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_msg = event.message.text
    print(f"收到訊息: {user_msg}") # 在終端機印出來讓你看
    
    # 回覆一樣的話 (Echo)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=f"K測試成功！真是Lucky啊！你說了：{user_msg}")
    )