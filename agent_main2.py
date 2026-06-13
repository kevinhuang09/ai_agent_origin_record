import requests
import re
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# ==========================================
# ⚠️ 請將這裡替換成你最新的 Cloudflare 網址
# ==========================================
BASE_TUNNEL_URL = "https://warm-himself-limitations-quit.trycloudflare.com"

print(f"🔧 正在連線至隧道: {BASE_TUNNEL_URL}")

llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url=f"{BASE_TUNNEL_URL}/ollama"
)

# ==========================================
# 🛡️ 防彈輸入清理函數
# ==========================================
def clean_sequence(input_val):
    """只保留字母，移除數字、逗號、引號、空格"""
    s = str(input_val)
    # 處理 LLM 可能丟出的 {"value": "..."} 格式
    if "value" in s:
        match = re.search(r"['\"]value['\"]:\s*['\"]([^'\"]+)['\"]", s)
        if match: s = match.group(1)
    
    clean = re.sub(r'[^a-zA-Z]', '', s).upper()
    return clean

# ==========================================
# 🛠️ 定義各項 AI 工具
# ==========================================

# ── 終極大招：全自動分析工具 ──
@tool
def full_auto_analysis_tool(sequence: str) -> str:
    """
    一鍵執行全自動化分析。
    包含：變體生成、挑選最優序列、理化繪圖、等電點計算、3D結構預測。
    """
    # ⚡ 強制清理輸入，防止 Agent 帶入 ",1" 導致 422 錯誤
    target_seq = clean_sequence(sequence)
    
    url = f"{BASE_TUNNEL_URL}/auto_research_pipeline"
    try:
        response = requests.post(url, json={"sequence": target_seq}, timeout=300)
        data = response.json()
        if data["status"] == "success":
            return (f"✅ 全自動分析完成！\n"
                    f"- 原始序列: {data['original_seq']}\n"
                    f"- 最優變體: {data['best_variant']} (分數: {data['variant_score']})\n"
                    f"- 理化性質: MW {data['physicochemical']['molecular_weight']}, pI {data['physicochemical']['isoelectric_point']}\n"
                    f"- 繪圖路徑: {data['physicochemical']['image_path']}\n"
                    f"- PDB路徑: {data['pdb_path']}\n"
                    f"請向使用者彙整以上數據。")
        return f"失敗：{data.get('message')}"
    except Exception as e:
        return f"連線異常: {str(e)}"

# ── 工具 1：神經肽預測 ──
@tool
def neuropeptide_predictor_tool(sequence: str) -> str:
    """用於分析單一蛋白質序列是否為神經肽。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/predict"
    try:
        print(f"📡 正在傳送序列 '{clean_seq}' 至 5090 進行單次預測...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "error": return f"伺服器報錯：{data.get('message')}"
        res = data.get("prediction", {})
        return (f"預測完畢！結果標籤為【{res.get('label')}】，信心分數為【{res.get('score')}】。\n"
                f"資料來源：【{res.get('note')}】。")
    except Exception as e: return f"❌ 工具呼叫失敗: {str(e)}"

# ── 工具 2：Ankh 特徵提取 ──
@tool
def ankh_protein_language_tool(sequence: str) -> str:
    """當使用者詢問蛋白質的深層特徵、Embedding 時使用。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/ankh"
    try:
        print(f"📡 正在請求 Ankh 引擎分析...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        if data.get("status") == "success": return f"Ankh 分析完成！\n特徵摘要：{data['result']}"
        return f"Ankh 工具報錯：{data.get('message')}"
    except Exception as e: return f"❌ 呼叫 Ankh 失敗: {str(e)}"

# ── 工具 3：變體生成 ──
@tool
def protein_discovery_pipeline_tool(query: str) -> str:
    """
    執行蛋白質發現與進化管線 (Discovery Pipeline)。
    ⚠️ 嚴格規定：請傳入「序列,數量」格式的純字串。
    例如如果要對 GPRLVRF 產生 10 條，請精準輸入：GPRLVRF,10
    絕對不要傳入 JSON 或字典格式！
    """
    # 這裡因為需要逗號分隔，所以稍微不同，我們特別處理
    clean_query = str(query).replace("'", "").replace('"', '').replace('{', '').replace('}', '').strip()
    parts = clean_query.split(',')
    
    clean_seq = clean_sequence(parts[0])
        
    if not clean_seq or len(clean_seq) < 2:
        return "❌ 錯誤：請提供正確的蛋白質序列。"
        
    num_variants = 100
    if len(parts) > 1:
        try:
            num_match = re.search(r'\d+', parts[1])
            if num_match:
                num_variants = int(num_match.group())
        except Exception:
            num_variants = 100

    url = f"{BASE_TUNNEL_URL}/discovery_pipeline" 
    try:
        print(f"🧬 啟動管線：針對序列 '{clean_seq}' 生成 {num_variants} 條變體並進行預測...")
        
        response = requests.post(url, json={"sequence": clean_seq, "num_variants": num_variants}, timeout=300) 
        data = response.json()
        
        if data.get("status") == "success":
            results = data.get("top_results", [])
            if not results:
                return "✅ 管線執行完畢，但未能找到任何有效的候選序列（可能被過濾或預測失敗）。"
                
            res_text = f"✅ 管線執行成功！\n- 生成變體要求: {data['total_generated']} 條\n- 去冗餘後剩餘: {data['after_cdhit']} 條\n\n前 {len(results)} 名高分候選序列 (已由 AI 模型打分)：\n"
            for i, r in enumerate(results, 1):
                res_text += f"{i}. 序列: {r['seq']} | 分數: {r['score']} | 標籤: {r['label']}\n"
            return res_text
            
        return f"❌ 伺服器回傳錯誤: {data.get('message')}"
    except requests.exceptions.Timeout:
        return "❌ 連線超時：伺服器運算超過 5 分鐘，可能仍在背景執行中。"
    except Exception as e:
        return f"❌ 連線失敗: {str(e)}"

# ── 工具 4：二級結構分析 ──
@tool
def secondary_structure_tool(sequence: str) -> str:
    """
    當使用者詢問蛋白質的「二級結構」(Secondary Structure)、「Alpha 螺旋 (Helix)」、「Beta 摺疊 (Sheet)」或「無規捲曲 (Coil)」時使用。
    請傳入純字串的蛋白質序列。
    """
    clean_seq = clean_sequence(sequence)
    if not clean_seq or len(clean_seq) < 2:
        return "❌ 錯誤：請提供正確的蛋白質序列。"

    url = f"{BASE_TUNNEL_URL}/secondary_structure"
    try:
        print(f"🧬 正在分析序列 '{clean_seq}' 的二級結構...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        
        if data.get("status") == "success":
            return (f"二級結構分析完成！\n"
                    f"序列: {data['sequence']}\n"
                    f"結構: {data['structure']}\n"
                    f"統計: {data['summary']}")
        return f"伺服器報錯：{data.get('message')}"
    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"

# ── 工具 5：ESMFold 3D 結構預測 ──
@tool
def esmfold_3d_structure_tool(sequence: str) -> str:
    """
    當使用者要求預測蛋白質的「3D 結構」、「立體結構」或「使用 ESMFold」時使用。
    請傳入純字串的蛋白質序列。
    注意：此工具會生成 PDB 檔案路徑。
    """
    clean_seq = clean_sequence(sequence)

    url = f"{BASE_TUNNEL_URL}/esmfold"
    try:
        print(f"🧬 正在啟動 ESMFold 預測 '{clean_seq}' 的 3D 結構...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        
        if data.get("status") == "success":
            return f"✅ 3D 結構預測大成功！\nPDB 檔案已安全儲存於伺服器路徑：【{data['file_path']}】。\n請告訴使用者可以使用 PyMOL 或 ChimeraX 等軟體開啟此檔案來觀看 3D 結構。"
        return f"伺服器報錯：{data.get('message')}"
    except requests.exceptions.Timeout:
        return "❌ 連線超時：3D 預測比較耗時，請稍後檢查伺服器。"
    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"

# ── 工具 6：理化性質與繪圖 ──
@tool
def protein_physicochemical_plot_tool(sequence: str) -> str:
    """
    當使用者要求「畫出理化性質圖」、「分析疏水性」或「計算等電點/分子量」時使用。
    輸入應為蛋白質序列。
    """
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/analyze_physicochemical"
    
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=30)
        data = response.json()
        
        if data["status"] == "success":
            return (f"📊 理化性質分析完成！\n"
                    f"- 分子量 (MW): {data['molecular_weight']}\n"
                    f"- 等電點 (pI): {data['isoelectric_point']}\n"
                    f"- 分析圖表已儲存至：【{data['image_path']}】\n"
                    f"圖表中的波峰代表疏水區域，波谷代表親水區域。這有助於了解肽鏈與膜的交互作用。")
        return f"錯誤：{data['message']}"
    except Exception as e:
        return f"呼叫工具失敗: {str(e)}"


# ==========================================
# 🤖 Agent 初始化區塊
# ==========================================

# 確保所有工具（包含全自動大招）都在清單內
tools = [
    full_auto_analysis_tool,
    neuropeptide_predictor_tool, 
    ankh_protein_language_tool, 
    protein_discovery_pipeline_tool, 
    secondary_structure_tool, 
    esmfold_3d_structure_tool, 
    protein_physicochemical_plot_tool
]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    max_iterations=10,  # 調高迭代次數，避免提早放棄
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": "你是一個自動化生物研究機器人。當使用者要求連鎖分析或自動化流程時，請『優先』使用 full_auto_analysis_tool 一鍵完成任務。"
    }
)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 遠端 Agent 指揮中心啟動完畢！")
    print("="*50)
    while True:
        user_input = input("\n請輸入指令 (exit 離開): ")
        if user_input.lower() in ['exit', 'quit']: break
        try: 
            agent.invoke({"input": user_input})
        except Exception as e: 
            print(f"\n❌ Agent 執行發生錯誤: {e}")