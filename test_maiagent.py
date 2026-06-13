import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MAIAGENT_API_KEY")

CHATBOT_ID = "ea6bff29-4f47-474e-9b63-d44d1af06b42"

url = f"https://api.maiagent.ai/api/v1/chatbots/{CHATBOT_ID}/completions/"
headers = {
    "Authorization": f"Api-Key {API_KEY}",
    "Content-Type": "application/json"
}

# ✅ message 改成 dict
payload = {
    "message": {
        "content": "你好！請用繁體中文自我介紹，並告訴我你是哪個模型。"
    }
}

print("🚀 送出訊息中...")
response = requests.post(url, headers=headers, json=payload, timeout=120)

print(f"\n📡 狀態碼：{response.status_code}")
print(f"📦 完整回應：\n{response.text[:2000]}")
