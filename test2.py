import requests

# 你的 ngrok 網址
URL = "https://pellicular-becki-nastily.ngrok-free.dev/predict"

def test_protein_prediction():
    # 測試序列
    test_seq = "MKVLVLYNGKTYKGEF" 
    
    print(f"📤 正在發送序列到 5090 進行預測...")
    response = requests.post(URL, json={"sequence": test_seq})
    
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "success":
            print("✅ 預測成功！結果如下：")
            print(result["prediction"])
        else:
            print(f"❌ 5090 報錯：{result['message']}")
    else:
        print(f"❌ 連線失敗，狀態碼：{response.status_code}")

if __name__ == "__main__":
    test_protein_prediction()
