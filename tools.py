from langchain.tools import tool
import requests

@tool
def neuropeptide_predictor_tool(sequence: str) -> str:
    """
    輸入一個蛋白質序列（例如: MASVMT...），
    此工具會呼叫 5090 伺服器上的 LightGBM 模型進行預測，
    並回傳該序列是神經肽的機率。
    """
    url = "http://你的ngrok網址/predict"  # 這裡填你之前的 ngrok 網址
    payload = {"sequence": sequence.strip().upper()}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            res = response.json()
            if res["status"] == "success":
                data = res["prediction"]
                return f"序列: {data['sequence']}\n預測機率: {data['probability']}\n模型: {data['model_used']}"
            else:
                return f"預測失敗: {res.get('message')}"
        return f"連線錯誤，狀態碼: {response.status_code}"
    except Exception as e:
        return f"發生錯誤: {str(e)}"

