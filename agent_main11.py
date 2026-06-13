import requests
import re
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import subprocess
from datetime import datetime
import os
import requests
import os
import requests
import subprocess
from datetime import datetime
from langchain.tools import tool
def generate_pymol_view_local(pdb_path: str, view_type: str) -> str:
    """本機 PyMOL 渲染核心：一次僅產出一張特定視圖，確保路徑純粹，避開 PDF 合併錯誤。"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%H%M%S')
    img_filename = f"local_{view_type}_{timestamp}.png"
    img_path = os.path.join(current_dir, img_filename)
    script_path = os.path.join(current_dir, f"temp_{view_type}.pml")
    
    with open(script_path, 'w') as f:
        f.write(f"load {pdb_path}, prot\nbg_color white\nhide all\n")
        if view_type == "ribbon":
            f.write("show cartoon, prot\ncolor spectrum, prot\n")
        elif view_type == "atoms":
            f.write("show sticks, prot\ncolor cpk, prot\n")
        elif view_type == "surface":
            f.write("show surface, prot\n")
            f.write("set_color hydrophob, [1.0, 0.5, 0.5]\n")
            f.write("color hydrophob, resn ALA+VAL+LEU+ILE+PHE+TRP+MET+PRO\n")
            f.write("color marine, resn ARG+LYS+HIS+ASP+GLU\n")
        f.write("zoom\nray 800, 600\n")
        f.write(f"png {img_path}\nquit\n")
        
    subprocess.run(["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path], check=True)
    if os.path.exists(script_path): os.remove(script_path)
    return img_path
def generate_pymol_plots_triple(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> list:
    """
    一口氣生成三種風格的圖片並附帶專業說明：
    1. Ribbons (二級結構)
    2. All Atoms (原子交互作用)
    3. Hydrophobicity Surface (蛋白質表面特性)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [下載 PDB 邏輯保持不變...]
    local_pdb_path = os.path.join(current_dir, f"temp_{file_prefix}.pdb")
    if not os.path.exists(local_pdb_path):
        # 這裡放入你之前的 requests 下載 PDB 程式碼
        pass

    # 定義圖片存檔路徑
    results = [
        {
            "title": "Interactive 1 (Ribbons) - 二級結構緞帶模型",
            "path": os.path.join(current_dir, f"render_{file_prefix}_1_ribbon.png"),
            "desc": "說明：此圖以緞帶模型展示蛋白質的二級結構（螺旋與折疊）。顏色按殘基順序呈彩虹漸層，有助於觀察整體的空間摺疊構型。"
        },
        {
            "title": "Interactive 2 (All Atoms) - 全原子棒狀模型",
            "path": os.path.join(current_dir, f"render_{file_prefix}_2_atoms.png"),
            "desc": "說明：此圖展示所有原子的棒狀模型。碳為灰色、氮為藍色、氧為紅色，適合觀察特定氨基酸側鏈的化學環境與交互作用。"
        },
        {
            "title": "Interactive 3 (Hydrophobicity Surface) - 親疏水表面模型",
            "path": os.path.join(current_dir, f"render_{file_prefix}_3_surface.png"),
            "desc": "說明：蛋白質表面根據親疏水性著色。紅色區域代表疏水性（Hydrophobic），藍色區域代表親水性（Hydrophilic），用於分析潛在的結合口袋。"
        }
    ]
    
    script_path = os.path.join(current_dir, f"render_triple.pml")
    
    with open(script_path, 'w') as f:
        f.write(f"load {local_pdb_path}, myprot\n")
        f.write("bg_color white\n") # PDF 建議白底
        
        # 繪製 Ribbon
        f.write("hide all\nshow cartoon, myprot\ncolor spectrum, myprot\n")
        f.write(f"png {results[0]['path']}, width=800, height=600, ray=1\n")
        
        # 繪製 Sticks
        f.write("hide all\nshow sticks, myprot\ncolor cpk, myprot\n")
        f.write(f"png {results[1]['path']}, width=800, height=600, ray=1\n")
        
        # 繪製 Surface
        f.write("hide all\nshow surface, myprot\n")
        f.write("set_color hydrophob, [1.0, 0.5, 0.5]\n") # 疏水淺紅
        f.write("color hydrophob, resn ALA+VAL+LEU+ILE+PHE+TRP+MET+PRO\n")
        f.write("color slate, resn ASN+GLN+SER+THR+TYR+CYS\n")
        f.write("color marine, resn ARG+LYS+HIS+ASP+GLU\n")
        f.write(f"png {results[2]['path']}, width=800, height=600, ray=1\n")
        f.write("quit\n")
        
    pymol_cmd = ["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path]
    subprocess.run(pymol_cmd, check=True, capture_output=True)
    
    return results
def generate_pymol_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    """
    在本機端呼叫 PyMOL 生成專業黑底棒狀模型圖 (Stick Model)。
    適用於 WSL/Ubuntu 無螢幕環境。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 取得 PDB 內容並存成本機暫存檔
    pdb_data = ""
    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except:
            pass
            
    if not pdb_data:
        print("⚠️ [系統] 讀取遠端 PDB 失敗，改用 API 獲取座標...")
        response = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence, timeout=30)
        pdb_data = response.text
        
    local_pdb_path = os.path.join(current_dir, f"temp_{file_prefix}.pdb")
    with open(local_pdb_path, 'w') as f:
        f.write(pdb_data)
        
    # 2. 設定圖片與指令腳本路徑
    img_filename = f"pymol_render_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    script_path = os.path.join(current_dir, f"render_{file_prefix}.pml")
    
    # 3. 建立 PyMOL 專用腳本 (還原你想要的黑底原子棒狀模型)
    with open(script_path, 'w') as f:
        f.write(f"load {local_pdb_path}, myprotein\n")
        f.write("hide all\n")                        # 隱藏預設顯示
        
        # --- 樣式設定 ---
        f.write("show sticks, myprotein\n")          # 顯示棒狀模型
        f.write("set stick_radius, 0.25\n")         # 調整棒子粗細
        f.write("set stick_ball, on\n")             # 在節點處加上小球，更有質感
        f.write("set stick_ball_ratio, 1.5\n")      # 球的大小比例
        
        # --- 顏色設定 (CPK 原子配色) ---
        f.write("color cpk, myprotein\n")           # 標準配色：N藍、O紅、S黃
        f.write("color gray70, elem c\n")           # 碳原子設為淺灰色
        
        # --- 環境與渲染設定 ---
        f.write("bg_color black\n")                 # 背景設為黑色
        f.write("set ray_shadows, 0\n")             # 關閉陰影讓線條更乾淨
        f.write("util.cbc myprotein\n")             # 按鏈變色(若有多條鏈時)
        f.write("zoom\n")                           # 自動對焦到分子中心
        
        # --- 輸出 ---
        f.write(f"ray 800, 600\n")                  # 高品質光線追蹤渲染
        f.write(f"png {img_path}\n")                # 儲存圖片
        f.write("quit\n")
        
    # =======================================================
    # 🟢 完美執行路徑：使用 xvfb-run 搭配 conda 環境內的 python 模組
    # =======================================================
    # 注意：這裡使用 "python -m pymol" 是為了確保抓到你的 Conda 環境 biomni_e1
    pymol_cmd = ["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path]
    
    try:
        print(f"⏳ [系統] 正在使用 PyMOL 渲染黑底棒狀模型...")
        # 捕捉輸出以便出錯時除錯
        result = subprocess.run(pymol_cmd, check=True, capture_output=True, text=True)
        
        # 4. 清理暫存腳本與 PDB
        if os.path.exists(script_path): os.remove(script_path)
        if os.path.exists(local_pdb_path): os.remove(local_pdb_path)
        
        print(f"✅ 圖片已生成: {img_path}")
        return img_path
        
    except subprocess.CalledProcessError as e:
        error_info = e.stderr if e.stderr else e.stdout
        raise Exception(f"PyMOL 渲染失敗。詳細資訊:\n{error_info}")
def generate_chimera_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    """在 Ubuntu 本機端呼叫原生 Chimera 1.19 搭配虛擬螢幕生成 3D 渲染圖"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 取得 PDB 內容並存成本機暫存檔
    pdb_data = ""
    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except:
            pass
            
    if not pdb_data:
        print("⚠️ [系統] 讀取遠端 PDB 失敗，改用 API 獲取座標...")
        response = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence, timeout=30)
        pdb_data = response.text
        
    local_pdb_path = os.path.join(current_dir, f"temp_{file_prefix}.pdb")
    with open(local_pdb_path, 'w') as f:
        f.write(pdb_data)
        
    # 2. 設定圖片與指令腳本路徑
    img_filename = f"chimera_render_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    cmd_path = os.path.join(current_dir, f"render_{file_prefix}.com")
    
    # 3. 建立 Chimera 指令腳本
    with open(cmd_path, 'w') as f:
        f.write(f"open {local_pdb_path}\n")
        f.write("preset apply pub 1\n")
        f.write("color rainbow chain\n")
        f.write(f"copy file {img_path} width 800 height 600 supersample 3\n")
        f.write("stop\n")
        
    # ==========================================
    # 🟢 完美 Linux 原生呼叫 (搭配 xvfb 虛擬螢幕)
    # ==========================================
    chimera_cmd = ["xvfb-run", "-a", "chimera", "--nogui", "--silent", cmd_path]
    
    try:
        print("⏳ [系統] 正在使用 Linux 原生 Chimera 渲染中...")
        subprocess.run(chimera_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 清理暫存檔
        if os.path.exists(cmd_path): os.remove(cmd_path)
        if os.path.exists(local_pdb_path): os.remove(local_pdb_path)
        
        return img_path
    except Exception as e:
        raise Exception(f"Chimera 執行失敗: {str(e)}")     
# 🟢 新增：負責在本機將 PDB 轉換為 3D 骨架圖的函數
# ==========================================
def generate_local_3d_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    """在本機端生成蛋白質 3D 骨架(Backbone)預覽圖"""
    pdb_data = ""
    
    # 1. 嘗試讀取伺服器產生的 PDB 檔 (如果權限允許的話)
    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except:
            pass
            
    # 2. 如果讀不到 (跨使用者權限問題)，使用 ESM 公開 API 即時獲取座標
    if not pdb_data:
        print("⚠️ [系統] 無法讀取遠端 PDB，改為透過 API 獲取 3D 座標...")
        # 呼叫 Meta 官方的 ESM API 獲取 PDB 結構
        response = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence, timeout=30)
        pdb_data = response.text

    # 3. 解析 PDB 格式中的 C-alpha (CA) 主鏈碳原子座標
    x, y, z = [], [], []
    for line in pdb_data.split('\n'):
        # 尋找 CA (C-alpha) 原子
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                # PDB 格式固定欄位擷取座標
                x.append(float(line[30:38]))
                y.append(float(line[38:46]))
                z.append(float(line[46:54]))
            except ValueError:
                pass

    # 4. 使用 matplotlib 畫出 3D 空間的骨架圖
    fig = plt.figure(figsize=(6, 6))
    # projection='3d' 讓畫布變成 3D 模式
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製連續的線條 (蛋白質骨架)
    ax.plot(x, y, z, color='mediumpurple', linewidth=2.5)
    # 繪製原子節點，增加立體感
    ax.scatter(x, y, z, color='darkslateblue', s=15)
    
    ax.set_title(f"Protein 3D Backbone Trace", fontsize=14)
    # 隱藏背景的座標軸網格，讓蛋白質看起來更乾淨像懸浮在空中
    ax.axis('off') 

    # 5. 儲存圖片到本機
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_filename = f"local_3d_plot_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    
    print(f"🎨 [系統] 本機 3D 預覽圖已生成: {img_path}")
    return img_path

# ==========================================
# ⚠️ 請將這裡替換成你最新的 Cloudflare 網址
# ==========================================
BASE_TUNNEL_URL = "https://maker-catalogs-flowers-earliest.trycloudflare.com"

print(f"🔧 正在連線至隧道: {BASE_TUNNEL_URL}")

from datetime import datetime
from fpdf import FPDF

# 全局日誌容器
GLOBAL_LOGS = []

def add_log(step_name, content):
    """將步驟紀錄加入全局日誌"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] --- {step_name} ---\n{content}\n"
    GLOBAL_LOGS.append(log_entry)
    print(f"📝 日誌已更新: {step_name}") # 在終端機也印出確認

import os
from fpdf import FPDF
from datetime import datetime

def sanitize_pdf_text(text: str) -> str:
    """將常見的 Emoji 替換為文字標籤，避免 Unicode 渲染報錯。"""
    replacements = {
        "✅": "[OK]", "❌": "[ERROR]", "📊": "[PLOT]",
        "🚀": "[START]", "📝": "[LOG]", "📡": "[SEND]",
        "🔧": "[CONFIG]", "⚠️": "[WARN]"
    }
    for emoji, label in replacements.items():
        text = text.replace(emoji, label)
    return text

import os
import re
from fpdf import FPDF
from datetime import datetime

@tool
def generate_pdf_report_tool(query: str = "生成報告") -> str:
    """彙整本次對話中所有工具的執行紀錄，生成最終 PDF 報告（支援自動插入圖片與防崩潰排版）。"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # 設定字體路徑 (請根據您本機實際路徑調整)
        font_path = "/home/tempadmin/agentic_ai/msjh.ttc" 
        
        has_unicode_font = False
        if os.path.exists(font_path):
            pdf.add_font("msjh", style="", fname=font_path)
            pdf.set_font("msjh", size=16)
            has_unicode_font = True
        else:
            pdf.set_font("helvetica", size=16)

        # 1. 寫入主標題
        pdf.cell(200, 10, text="Protein Research Comprehensive Report", new_x="LMARGIN", new_y="NEXT", align='C')
        
        if has_unicode_font:
            pdf.set_font("msjh", size=10)
        else:
            pdf.set_font("helvetica", size=10)
            
        pdf.ln(10)
        
        if not GLOBAL_LOGS:
            pdf.multi_cell(0, 8, text="No logs recorded in this session.")
        else:
            raw_content = "\n".join(GLOBAL_LOGS)
            safe_content = sanitize_pdf_text(raw_content)
            
            if not has_unicode_font:
                safe_content = safe_content.encode('latin-1', 'replace').decode('latin-1')
            
            seen_images = set()
            lines = safe_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    pdf.ln(2)
                    continue
                
                line = re.sub(r'-{15,}', '--------------------', line)
                
                if pdf.get_y() > 250:
                    pdf.add_page()
                
                pdf.set_x(pdf.l_margin)
                
                try:
                    pdf.multi_cell(0, 7, text=line)
                except Exception:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, text=line[:60] + " ... (protect)")
                
                png_match = re.search(r'(/[\w\./_-]+\.png)', line)
                if png_match:
                    img_path = png_match.group(1)
                    if os.path.exists(img_path) and img_path not in seen_images:
                        pdf.ln(2)
                        try:
                            pdf.image(img_path, w=140)
                            seen_images.add(img_path)
                            pdf.ln(5)
                        except Exception as img_e:
                            pdf.multi_cell(0, 8, text=f"[Image Load Failed]")

        report_name = f"Full_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(report_name)
        return f"✅ 完整報告已生成：{report_name}"
        
    except Exception as e:
        return f"❌ PDF 生成失敗: {str(e)}"

# ==========================================
# 🤖 重要：請確保這裡有換行，不要黏在上面的 return 
# ==========================================

llm = ChatOllama(
    model="gemma2:9b", # 或是您使用的模型名稱
    temperature=0,
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
    """一鍵執行全自動化分析。輸入格式為 '序列,數量'。"""
    parts = str(query).split(',')
    target_seq = clean_sequence(parts[0])
    num_variants = int(re.search(r'\d+', parts[1]).group()) if len(parts) > 1 else 10
    url = f"{BASE_TUNNEL_URL}/auto_research_pipeline"
    try:
        response = requests.post(url, json={"sequence": target_seq, "num_variants": num_variants}, timeout=600)
        data = response.json()
        
        if data["status"] == "success":
            best_seq = data['best_variant']
            
            # ==========================================
            # 🟢 魔法發生在這裡：自己畫圖並取得本機路徑！
            # ==========================================
            local_img_path = generate_local_plot(best_seq, "BestVariant")
            
            # 將結果整理存入日誌，注意這裡的繪圖路徑改成我們的 local_img_path
            res_summary = (f"原始序列: {data['original_seq']}\n"
                           f"最優變體: {best_seq}\n"
                           f"MW: {data['physicochemical']['molecular_weight']}\n"
                           f"pI: {data['physicochemical']['isoelectric_point']}\n"
                           f"本機繪圖路徑: {local_img_path}\n"  # <--- PDF 生成工具會抓這行！
                           f"PDB路徑: {data['pdb_path']}")
            
            add_log("Full Auto Analysis", res_summary)
            return f"[OK] 全自動分析完成！\n{res_summary}"
            
        return f"[ERROR] 分析失敗：{data.get('message')}"
    except Exception as e: 
        return f"異常: {str(e)}"

# ── 工具 1：神經肽預測 ──
@tool
def neuropeptide_predictor_tool(sequence: str) -> str:
    """用於分析單一蛋白質序列是否為神經肽。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/predict"
    try:
        print(f"📡 正在傳送序列 '{clean_seq}' 至 Server 進行單次預測...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        
        if data.get("status") == "error": 
            return f"伺服器報錯：{data.get('message')}"
        
        res = data.get("prediction", {})
        
        # 1. 先組合要記錄與回傳的文字內容 (解決 result_text 未定義問題)
        result_text = (
            f"序列: {clean_seq}\n"
            f"預測結果: 【{res.get('label')}】\n"
            f"預測機率: 【{res.get('score', '0.0')}】\n"
            f"資料來源: 【{res.get('note')}】"
        )
        
        # 2. 將內容加入全局日誌 (這樣 PDF 才會抓到這段)
        add_log("Neuropeptide Prediction", result_text)
        
        # 3. 回傳給 Agent 顯示在畫面上的文字
        return f"預測完畢！\n{result_text}"
        
    except Exception as e: 
        return f"❌ 工具呼叫失敗: {str(e)}"

# ── 工具 2：Ankh 特徵提取 ──
@tool
def ankh_protein_language_tool(sequence: str) -> str:
    """
    輸入單一蛋白質序列進行深層特徵分析。
    請注意：一次只能傳入「一條」序列（例如：評分最高的 Variant）。
    """
    clean_seq = clean_sequence(sequence)
    # 防呆機制：如果 Agent 還是傳了多條序列（包含逗號或太長），截取第一部分
    if ',' in clean_seq:
        clean_seq = clean_seq.split(',')[0]
    
    url = f"{BASE_TUNNEL_URL}/ankh_feature"
    try:
        # 在發送請求前印出偵錯資訊，確保序列正確
        print(f"📡 正在請求 Ankh 引擎分析單一序列: '{clean_seq}'...")
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        
        if data.get("status") == "success":
            result_text = f"Variant 序列: {clean_seq}\nAnkh 特徵摘要：{data.get('result', '分析完成')}"
            add_log("Ankh Protein Language Tool", result_text)
            return f"✅ Ankh 分析完成！\n{result_text}"
        
        return f"❌ Ankh 工具報錯：{data.get('message', '未知錯誤')}"
    except Exception as e: 
        return f"❌ 呼叫 Ankh 失敗: {str(e)}"
# ── 工具 3：變體生成 ──

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
            result_text = (f"二級結構分析完成！\n序列: {data['sequence']}\n結構: {data['structure']}\n統計: {data['summary']}")
            add_log("Secondary Structure Analysis", result_text)
            return result_text
        error_msg = f"伺服器報錯：{data.get('message')}"
        add_log("Secondary Structure Analysis (Error)", error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"❌ 工具呼叫失敗: {str(e)}"
        add_log("Secondary Structure Analysis (Exception)", error_msg)
        return error_msg

# ── 工具 5：ESMFold 3D 結構預測 ──
# ==========================================
# 🟢 替換：更新後的 3D 結構預測工具
# ==========================================
# ── 工具 5：ESMFold 3D 結構預測 (PyMOL 升級版) ──
# ── 修改後的工具 3：蛋白質序列優化 (Sequence Optimization) ──
@tool
def protein_sequence_optimization_tool(query: str) -> str:
    """
    執行蛋白質序列優化 (Sequence Optimization)。
    輸入格式為單一字串，以逗號分隔參數: '序列,位置,比例'
    例如: 'RLLGVLIPCRAFGDV,3 7 11,0.1'
    - 序列: 原始胺基酸序列
    - 位置: 指定要優化的位置(以空格隔開)，若全優化則填 'all'
    - 比例: 突變百分比 (0.0 ~ 1.0)
    """
    try:
        # 1. 強制解析單一字串 query
        params = [p.strip() for p in str(query).split(',')]
        if not params[0] or len(params[0]) < 2:
            return "❌ 錯誤：請提供正確的序列格式。"
            
        sequence = clean_sequence(params[0])
        target_positions = params[1] if len(params) > 1 else "all"
        
        try:
            # 解析 Mutation Rate，預設為 0.1
            mutation_rate = float(params[2]) if len(params) > 2 else 0.1
        except ValueError:
            mutation_rate = 0.1
        
        # 2. 連線設定 (確認 Port 為 8800)
        # 如果你全域有定義 BASE_TUNNEL_URL，會直接使用，否則這裡手動指定
        api_url = f"http://127.0.0.1:8800/discovery_pipeline"
        
        payload = {
            "sequence": sequence, 
            "target_positions": target_positions, 
            "mutation_rate": mutation_rate,
            "num_variants": 5
        }
        
        print(f"📡 正在發送序列優化請求至: {api_url}...")
        
        # 3. 發送請求
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status() 
        data = response.json()
        
        if data.get("status") == "success":
            results = data.get("top_results", [])
            
            # 使用專業術語：Sequence Optimization / Variant
            res_text = (f"✅ 序列優化 (Sequence Optimization) 完成\n"
                        f"- 原始序列: {sequence}\n"
                        f"- 設定突變比例 (Mutation Rate): {mutation_rate*100}%\n"
                        f"- 優化位置 (Target Positions): {target_positions}\n\n"
                        f"建議之高評分 Variant 列表：\n")
            
            for i, r in enumerate(results, 1):
                res_text += f"{i}. Variant: {r['seq']} | 分數: {r.get('score', 'N/A')} | 狀態: {r.get('label', 'N/A')}\n"
            
            # 存入日誌供 PDF 工具抓取
            add_log("Sequence Optimization", res_text)
            return res_text
        else:
            return f"❌ 伺服器邏輯錯誤: {data.get('message', '未知錯誤')}"

    except requests.exceptions.ConnectionError:
        return f"❌ 連線失敗：無法連接至伺服器。請確認 server.py 是否在 8800 埠執行。"
    except Exception as e:
        return f"❌ 執行出錯: {str(e)}"
        
def esmfold_3d_structure_tool(sequence: str) -> str:
    """預測 3D 結構。拿到 PDB 路徑後，請依序呼叫 render_protein_view_tool 生成視圖。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/esmfold"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "success":
            pdb_path = data['file_path']
            res = f"✅ 3D 預測成功！PDB：{pdb_path}。請接著渲染 ribbon, atoms, surface 視圖。"
            add_log("ESMFold Prediction", res)
            return res
        return f"失敗：{data.get('message')}"
    except Exception as e: return f"❌ 異常: {str(e)}"

# ── 拆分後的 3D 工具 5-2：本機分段渲染 (解決 MERG 報錯關鍵) ──
@tool
def render_protein_view_tool(query: str) -> str:
    """
    在本機渲染特定 3D 視圖。輸入格式為 'PDB路徑,視圖類型'。
    視圖類型必須是: 'ribbon'、'atoms' 或 'surface'。
    """
    try:
        params = [p.strip() for p in str(query).split(',')]
        pdb_path = params[0]
        view_type = params[1]
        
        titles = {
            "ribbon": "Interactive 1 (ribbons) - 二級結構緞帶模型",
            "atoms": "Interactive 2 (all atoms) - 全原子棒狀模型",
            "surface": "Interactive 3 (hydrophobicity surface) - 親疏水表面分析"
        }
        
        if view_type not in titles:
            return f"❌ 不支援的視圖類型: {view_type}"

        path = generate_pymol_view_local(pdb_path, view_type)
        log_content = f"### {titles[view_type]}\n- 本機繪圖路徑: {path}"
        add_log(f"3D Render {view_type.capitalize()}", log_content)
        return f"✅ {view_type} 渲染完成：{path}"
    except Exception as e:
        return f"❌ 渲染失敗: {str(e)}。請確保格式為 'PDB路徑,視圖類型'"
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
            # 💡 魔法在這裡：呼叫筆電自己畫圖，並取得本機的圖片路徑！
            local_img_path = generate_local_plot(clean_seq, "Physico")
            
            # 將本機路徑寫入 result_text，讓 PDF 工具的掃描器可以抓到它！
            result_text = (f"📊 理化性質分析完成！\n"
                           f"- 分子量 (MW): {data['molecular_weight']}\n"
                           f"- 等電點 (pI): {data['isoelectric_point']}\n"
                           f"- 遠端原圖路徑: {data['image_path']}\n"
                           f"- 本機繪圖路徑: {local_img_path}") # PDF 工具會抓這一行！
            
            add_log("Protein Physicochemical Plot Tool", result_text)
            return result_text
            
        error_msg = f"錯誤：{data['message']}"
        add_log("Protein Physicochemical Plot Tool (Error)", error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"呼叫工具失敗: {str(e)}"
        add_log("Protein Physicochemical Plot Tool (Exception)", error_msg)
        return error_msg
# ==========================================
# 🤖 Agent 初始化區塊
# ==========================================
tools = [
    full_auto_analysis_tool,
    neuropeptide_predictor_tool, 
    ankh_protein_language_tool, 
    protein_sequence_optimization_tool, # 確保這是一個被 @tool 裝飾的函數
    secondary_structure_tool, 
    esmfold_3d_structure_tool, 
    render_protein_view_tool, 
    protein_physicochemical_plot_tool,
    generate_pdf_report_tool
]

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    # agent_kwargs={
    #     "prefix": "你是一個自動化生物研究機器人。當使用者要求連鎖分析時，請優先使用 full_auto_analysis_tool。"
    # }
    agent_kwargs={
    "prefix": """你是一個專業的生物資訊自動化機器人。
    當完成序列優化 (Sequence Optimization) 後：
    1. 從建議清單中挑選「評分最高 (Top 1)」的 Variant。
    2. 將該單一序列傳入 ankh_protein_language_tool 進行特徵分析。
    3. 嚴禁一次將多條序列串聯傳送給 Ankh 工具。
    4. 最後依照 PDB 路徑進行 3D 渲染並生成報告。"""
}
)

if __name__ == "__main__":
    print("\n🚀 Agent 指揮中心啟動！")
    while True:
        user_input = input("\n請輸入指令 (exit 離開): ")
        if user_input.lower() in ['exit', 'quit']: break
        try:
            agent.invoke({"input": user_input})
            pdf_result = generate_pdf_report_tool.invoke("")
            print(f"\n{pdf_result}")
        except Exception as e:
            print(f"\n❌ 錯誤: {e}")

