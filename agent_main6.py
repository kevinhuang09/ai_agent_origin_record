import requests
import re
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# ==========================================
# ⚠️ 請將這裡替換成你最新的 Cloudflare 網址
# ==========================================
BASE_TUNNEL_URL = "https://loading-indicators-cross-anonymous.trycloudflare.com"

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
def full_auto_analysis_tool(query: str) -> str:
    """
    一鍵執行全自動化分析。輸入格式為 '序列,數量' (例如 GPRLVRF,500)。
    包含：變體生成、挑選最優序列、理化繪圖、等電點計算、3D結構預測。
    """
    # 解析輸入：處理可能是 "GPRLVRF" 或 "GPRLVRF,500" 的情況
    parts = str(query).split(',')
    target_seq = clean_sequence(parts[0])
    
    num_variants = 10 # 預設值
    if len(parts) > 1:
        try:
            # 提取數字
            num_match = re.search(r'\d+', parts[1])
            if num_match: num_variants = int(num_match.group())
        except: pass

    url = f"{BASE_TUNNEL_URL}/auto_research_pipeline"
    try:
        # 修改：將 num_variants 傳送給 Server
        payload = {"sequence": target_seq, "num_variants": num_variants}
        response = requests.post(url, json=payload, timeout=600) # 500條需要較長超時
        data = response.json()
        if data["status"] == "success":
            return (f"✅ 全自動分析完成！\n"
                    f"🔬 [Discovery] 從 {data['original_seq']} 衍生 {data.get('total_generated', 10)} 條序列...\n"
                    f"🧬 [Discovery] 去冗餘後剩餘 {data.get('after_cdhit', '未知')} 條，將序列打包進行預測...\n"
                    f"--------------------------------------------------\n"
                    f"- 原始序列: {data['original_seq']}\n"
                    f"- 最優變體: {data['best_variant']} (分數: {data.get('variant_score', '未知')})\n"
                    f"- 理化性質: MW {data['physicochemical']['molecular_weight']}, pI {data['physicochemical']['isoelectric_point']}\n"
                    f"- 繪圖路徑: {data['physicochemical']['image_path']}\n"
                    f"- PDB路徑: {data['pdb_path']}\n"
                    f"請向使用者彙整以上詳細發現過程與數據。")
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
        # 確保抓取 score 而不是 probability，因為 Server 現在會補全 Key
        return (f"預測完畢！結果標籤為【{res.get('label')}】，預測機率為【{res.get('score', '0.0')}】。\n"
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
    傳入格式：序列,數量 (例如 GPRLVRF,10)
    """
    clean_query = str(query).replace("'", "").replace('"', '').replace('{', '').replace('}', '').strip()
    parts = clean_query.split(',')
    clean_seq = clean_sequence(parts[0])
        
    if not clean_seq or len(clean_seq) < 2:
        return "❌ 錯誤：請提供正確的蛋白質序列。"
        
    num_variants = 100
    if len(parts) > 1:
        try:
            num_match = re.search(r'\d+', parts[1])
            if num_match: num_variants = int(num_match.group())
        except Exception: num_variants = 100

    url = f"{BASE_TUNNEL_URL}/discovery_pipeline" 
    try:
        print(f"🧬 啟動管線：針對序列 '{clean_seq}' 生成 {num_variants} 條變體並進行預測...")
        response = requests.post(url, json={"sequence": clean_seq, "num_variants": num_variants}, timeout=300) 
        data = response.json()
        
        if data.get("status") == "success":
            results = data.get("top_results", [])
            if not results: return "✅ 管線執行完畢，但未能找到任何候選序列。"
                
            res_text = f"✅ 管線執行成功！\n- 生成變體要求: {data['total_generated']} 條\n- 去冗餘後剩餘: {data['after_cdhit']} 條\n\n前 {len(results)} 名高分候選序列：\n"
            for i, r in enumerate(results, 1):
                # 抓取分數
                res_text += f"{i}. 序列: {r['seq']} | 分數: {r.get('score', r.get('probability'))} | 標籤: {r['label']}\n"
            return res_text
        return f"❌ 伺服器回傳錯誤: {data.get('message')}"
    except Exception as e: return f"❌ 連線失敗: {str(e)}"

# ── 工具 4：二級結構分析 ──
@tool
def secondary_structure_tool(sequence: str) -> str:
    """分析二級結構 (Helix/Sheet/Coil)。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/secondary_structure"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        if data.get("status") == "success":
            return (f"二級結構分析完成！\n序列: {data['sequence']}\n結構: {data['structure']}\n統計: {data['summary']}")
        return f"伺服器報錯：{data.get('message')}"
    except Exception as e: return f"❌ 工具呼叫失敗: {str(e)}"

# ── 工具 5：ESMFold 3D 結構預測 ──
@tool
def esmfold_3d_structure_tool(sequence: str) -> str:
    """預測立體結構。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/esmfold"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "success":
            return f"✅ 3D 結構預測大成功！\nPDB 檔案儲存於：【{data['file_path']}】。"
        return f"伺服器報錯：{data.get('message')}"
    except Exception as e: return f"❌ 工具呼叫失敗: {str(e)}"

# ── 工具 6：理化性質與繪圖 ──
@tool
def protein_physicochemical_plot_tool(sequence: str) -> str:
    """疏水性、等電點與分子量分析。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/analyze_physicochemical"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=30)
        data = response.json()
        if data["status"] == "success":
            return (f"📊 理化性質分析完成！\n"
                    f"- 分子量 (MW): {data['molecular_weight']}\n"
                    f"- 等電點 (pI): {data['isoelectric_point']}\n"
                    f"- 分析圖表儲存至：【{data['image_path']}】")
        return f"錯誤：{data['message']}"
    except Exception as e: return f"呼叫工具失敗: {str(e)}"

# ==========================================
# 🤖 Agent 初始化區塊
# ==========================================
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
    max_iterations=10,
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": "你是一個自動化生物研究機器人。當使用者要求連鎖分析時，請優先使用 full_auto_analysis_tool。"
    }
)

if __name__ == "__main__":
    print("\n🚀 Agent 指揮中心啟動！")
    while True:
        user_input = input("\n請輸入指令 (exit 離開): ")
        if user_input.lower() in ['exit', 'quit']: break
        try: agent.invoke({"input": user_input})
        except Exception as e: print(f"\n❌ 錯誤: {e}")

