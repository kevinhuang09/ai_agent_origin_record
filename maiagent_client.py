"""
MaiAgent API Client - 封裝呼叫邏輯
"""
import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class MaiAgentClient:
    """MaiAgent API 客戶端"""
    
    BASE_URL = "https://api.maiagent.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, chatbot_id: str = None):
        self.api_key = api_key or os.getenv("MAIAGENT_API_KEY")
        if not self.api_key:
            raise ValueError("❌ 請設定 MAIAGENT_API_KEY 環境變數")
        
        self.chatbot_id = chatbot_id
        self.conversation_id: Optional[str] = None  # 自動延續對話
        
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(
        self, 
        message: str, 
        new_conversation: bool = False,
        timeout: int = 120
    ) -> str:
        """
        送出訊息並取得回覆
        
        Args:
            message: 要送的訊息
            new_conversation: True 則開啟新對話（不延續歷史）
            timeout: 逾時秒數
        
        Returns:
            Claude 的回覆文字
        """
        if new_conversation:
            self.conversation_id = None
        
        url = f"{self.BASE_URL}/chatbots/{self.chatbot_id}/completions/"
        
        payload = {
            "message": {
                "content": message
            }
        }
        # 若有現存對話 ID 則延續對話
        if self.conversation_id:
            payload["conversation"] = self.conversation_id
        
        response = requests.post(
            url, 
            headers=self.headers, 
            json=payload, 
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"❌ MaiAgent API 錯誤 [{response.status_code}]: {response.text}"
            )
        
        data = response.json()
        # 記錄對話 ID 以便下次延續
        self.conversation_id = data.get("conversationId")
        
        return data.get("content", "")
    
    def reset(self):
        """重置對話歷史"""
        self.conversation_id = None
    
    def list_chatbots(self) -> list:
        """列出所有可用的 chatbot"""
        url = f"{self.BASE_URL}/chatbots/"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json().get("results", [])


# ============================================================
# 🧪 測試用 main
# ============================================================
if __name__ == "__main__":
    # NeuroP - Claude 4.7 Opus
    NEUROP_ID = "ea6bff29-4f47-474e-9b63-d44d1af06b42"
    
    client = MaiAgentClient(chatbot_id=NEUROP_ID)
    
    print("🤖 已連線到 NeuroP（Claude 4.7 Opus）")
    print("=" * 60)
    
    # 第一輪
    reply = client.chat("你好！我有一段序列：GPRLVRF，請幫我分析。")
    print(f"\n👤 你：你好！我有一段序列：GPRLVRF，請幫我分析。")
    print(f"\n🤖 NeuroP：\n{reply}")
    
    print("\n" + "=" * 60)
    
    # 第二輪 - 會自動延續對話
    reply = client.chat("剛剛那個序列的分類結果是什麼？")
    print(f"\n👤 你：剛剛那個序列的分類結果是什麼？")
    print(f"\n🤖 NeuroP：\n{reply}")
