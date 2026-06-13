import requests
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# ==========================================
# ⚠️ 請將這裡替換成你最新的 Cloudflare 網址
# 注意：結尾不要有斜線 (/)
# ==========================================
BASE_TUNNEL_URL = "https://was-way-greeting-moisture.trycloudflare.com"

print(f"🔧 正在連線至隧道: {BASE_TUNNEL_URL}")

# 1. 大腦：連線至代理伺服器的 /ollama 路徑
llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url=f"{BASE_TUNNEL_URL}/ollama"
)

# 2. 工具：連線至伺服器的 /predict 路徑
@tool
def neuropeptide_predictor_tool(sequence: str) -> str:
    """
    用於分析蛋白質序列是否為神經肽。
    請傳入純氨基酸字串（例如：AACT）。絕對不要傳入 JSON 格式。
    此工具僅限於「單一序列」預測。
    """
    if isinstance(sequence, dict):
        sequence = sequence.get("value", sequence.get("sequence", str(sequence)))
        
    clean_seq = str(sequence).strip().upper()
    url = f"{BASE_TUNNEL_URL}/predict"
    
    try:
        print(f"📡 正在傳送序列 '{clean_seq}' 至 5090 進行分析...")
        response = requests.post(
            url, 
            json={"sequence": clean_seq}, 
            timeout=120  
        )
        
        if response.status_code != 200:
            return f"❌ 伺服器錯誤，狀態碼: {response.status_code}"
            
        data = response.json()
        if data.get("status") == "error":
            return f"伺服器報錯：{data.get('message')}"

        res = data.get("prediction", {})
        label = res.get("label", "未知標籤")
        score = res.get("probability", "未知機率")
        note = res.get("note", "未知來源")
        
        return (f"預測完畢！結果標籤為【{label}】，預測機率為【{score}】。\n"
                f"資料來源/分析方法：【{note}】。\n"
                f"請立即使用 'Final Answer: ' 開頭回答使用者，並務必在回答中說明這個結果是來自 CD-HIT 還是 AI 管線計算！")

    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"

@tool
def ankh_protein_language_tool(sequence: str) -> str:
    """
    當使用者詢問蛋白質的深層特徵、Embedding、或是需要使用預訓練語言模型分析時使用。
    """
    if isinstance(sequence, dict):
        sequence = sequence.get("value", str(sequence))
    
    clean_seq = str(sequence).strip().upper()
    url = f"{BASE_TUNNEL_URL}/ankh"
    
    try:
        print(f"📡 正在請求 Ankh 引擎分析...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        
        if data.get("status") == "success":
            return (f"Ankh 模型分析完成！\n"
                    f"特徵摘要：{data['result']}\n"
                    f"這代表該序列已被轉化為高維空間中的數學特徵，可用於後續聚類或功能預測。")
        else:
            return f"Ankh 工具報錯：{data.get('message')}"
    except Exception as e:
        return f"❌ 呼叫 Ankh 失敗: {str(e)}"

@tool
def protein_discovery_pipeline_tool(sequence: str) -> str:
    """
    執行蛋白質發現與進化管線 (Discovery Pipeline)。
    當使用者要求生成變體、衍生序列、或批量篩選最高分候選序列時使用。
    請務必傳入有效的蛋白質序列（例如：GPRLVRF）。
    """
    if isinstance(sequence, dict):
        sequence = sequence.get("value", sequence.get("sequence", str(sequence)))
        
    clean_seq = str(sequence).strip().upper()
    
    # 🚨 前端防禦：阻擋 Agent 傳入空值或無效值
    if not clean_seq or clean_seq == "NONE" or len(clean_seq) < 2:
        return "❌ 錯誤：請提供正確的蛋白質序列（如 GPRLVRF），絕對不要傳入空值或過短序列。"
    
    url = f"{BASE_TUNNEL_URL}/discovery_pipeline" 
    try:
        print(f"🧬 正在啟動 Discovery 管線，針對序列 '{clean_seq}' 進行進化篩選...")
        # 🚨 批量運算耗時較長，Timeout 設為 300 秒 (5分鐘)
        response = requests.post(url, json={"sequence": clean_seq}, timeout=300) 
        data = response.json()
        
        if data["status"] == "success":
            results = data["top_results"]
            if not results:
                return "✅ 管線執行完畢，但未能找到任何有效的候選序列（可能被 CD-HIT 全數過濾或預測失敗）。"
                
            res_text = f"✅ 管線執行成功！\n- 生成變體: {data['total_generated']} 條\n- 去冗餘後剩餘: {data['after_cdhit']} 條\n\n前 {len(results)} 名高分候選序列：\n"
            for i, r in enumerate(results, 1):
                res_text += f"{i}. 序列: {r['seq']} | 分數: {r['score']} | 標籤: {r['label']}\n"
            return res_text
            
        return f"❌ 伺服器回傳錯誤: {data.get('message')}"
    except requests.exceptions.Timeout:
        return "❌ 連線超時：伺服器運算超過 5 分鐘，可能仍在背景執行中。"
    except Exception as e:
        return f"❌ 連線失敗: {str(e)}"

# 3. 初始化 Agent
tools = [neuropeptide_predictor_tool, ankh_protein_language_tool, protein_discovery_pipeline_tool]
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    max_iterations=5,  # 🚨 允許 Agent 思考更多步驟，避免提早放棄
    handle_parsing_errors=True
)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 遠端 Agent 指揮中心啟動完畢！")
    print("你可以輸入：'對序列 GPRLVRF 執行 Discovery Pipeline'")
    print("="*50)
    
    while True:
        user_input = input("\n請輸入指令 (exit 離開): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        try:
            agent.invoke({"input": user_input})
        except Exception as e:
            print(f"\n❌ Agent 執行時發生錯誤: {e}")