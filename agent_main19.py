import requests
import re
import os
import shutil
import customtkinter as ctk
import subprocess
import time
import threading
import queue
import sys
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from langchain_ollama import ChatOllama

# 🟢 修正：兼容多版本 LangChain 的 import 寫法（三層 fallback）
try:
    from langchain.agents import initialize_agent, AgentType
except ImportError:
    try:
        from langchain.agents.initialize import initialize_agent
        from langchain.agents.agent_types import AgentType
    except ImportError:
        from langchain.agents.initialize import initialize_agent
        from langchain.agents import AgentType

from langchain.tools import tool, Tool
from fpdf import FPDF

# === GUI 相關 ===
import tkinter as tk
from tkinter import ttk, scrolledtext, font as tkfont
from PIL import Image, ImageTk

# ==========================================
# ⚙️ 基本設定
# ==========================================
BASE_TUNNEL_URL = "https://rehabilitation-miami-printer-cells.trycloudflare.com"
print(f"🔧 正在連線至隧道: {BASE_TUNNEL_URL}")

GLOBAL_LOGS = []

def add_log(step_name, content):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] --- {step_name} ---\n{content}\n"
    GLOBAL_LOGS.append(log_entry)
    print(f"📝 日誌已更新: {step_name}")

def sanitize_pdf_text(text: str) -> str:
    replacements = {
        "✅": "[OK]", "❌": "[ERROR]", "📊": "[PLOT]",
        "🚀": "[START]", "📝": "[LOG]", "📡": "[SEND]",
        "🔧": "[CONFIG]", "⚠️": "[WARN]", "🧬": "[DNA]",
        "🔬": "[LAB]", "🎨": "[ART]", "🖼️": "[IMG]",
        "🛡️": "[SHIELD]", "🟢": "[GREEN]", "⏳": "[WAIT]",
        "⏱️": "[TIME]", "🩸": "[BLOOD]"
    }
    for emoji, label in replacements.items():
        text = text.replace(emoji, label)
    return text

def clean_sequence(input_val):
    s = str(input_val)
    if "value" in s:
        match = re.search(r"['\"]value['\"]:\s*['\"]([^'\"]+)['\"]", s)
        if match: s = match.group(1)
    clean = re.sub(r'[^a-zA-Z]', '', s).upper()
    return clean

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes} 分 {secs:.2f} 秒"

# ==========================================
# 🎨 本機繪圖工具區塊
# ==========================================
def generate_local_plot(sequence: str, file_prefix: str = "plot") -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        counter = Counter(sequence)
        aas = sorted(counter.keys())
        counts = [counter[a] for a in aas]

        plt.figure(figsize=(8, 4))
        plt.bar(aas, counts, color='steelblue', edgecolor='black')
        plt.title(f"Amino Acid Composition - {sequence[:15]}...")
        plt.xlabel("Amino Acid")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        img_filename = f"local_plot_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
        img_path = os.path.join(current_dir, img_filename)
        plt.savefig(img_path, dpi=150)
        plt.close()

        print(f"🎨 [系統] 本機分析圖已生成: {img_path}")
        return img_path
    except Exception as e:
        print(f"❌ 本機繪圖失敗: {e}")
        return f"繪圖失敗:{e}"


def _download_pdb_to_local(sequence: str, remote_pdb_path: str, file_prefix: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_data = ""

    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except Exception:
            pass

    if not pdb_data:
        print("⚠️ [系統] 讀取遠端 PDB 失敗，改用 ESM API 獲取座標...")
        response = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=sequence, timeout=60
        )
        pdb_data = response.text

    local_pdb_path = os.path.join(current_dir, f"temp_{file_prefix}.pdb")
    with open(local_pdb_path, 'w') as f:
        f.write(pdb_data)
    return local_pdb_path


def generate_pymol_plots_triple(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> list:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_pdb_path = _download_pdb_to_local(sequence, remote_pdb_path, file_prefix)

    results = [
        {"title": "Interactive 1 (Ribbons)", "path": os.path.join(current_dir, f"render_{file_prefix}_1_ribbon.png"), "desc": "緞帶模型"},
        {"title": "Interactive 2 (All Atoms)", "path": os.path.join(current_dir, f"render_{file_prefix}_2_atoms.png"), "desc": "全原子棒狀模型"},
        {"title": "Interactive 3 (Hydrophobicity)", "path": os.path.join(current_dir, f"render_{file_prefix}_3_surface.png"), "desc": "親疏水表面模型"}
    ]

    script_path = os.path.join(current_dir, f"render_triple.pml")
    with open(script_path, 'w') as f:
        f.write(f"load {local_pdb_path}, myprot\n")
        f.write("bg_color white\nhide all\nshow cartoon, myprot\ncolor spectrum, myprot\n")
        f.write(f"png {results[0]['path']}, width=800, height=600, ray=1\n")
        f.write("hide all\nshow sticks, myprot\ncolor cpk, myprot\n")
        f.write(f"png {results[1]['path']}, width=800, height=600, ray=1\n")
        f.write("hide all\nshow surface, myprot\n")
        f.write("set_color hydrophob, [1.0, 0.5, 0.5]\n")
        f.write("color hydrophob, resn ALA+VAL+LEU+ILE+PHE+TRP+MET+PRO\n")
        f.write("color slate, resn ASN+GLN+SER+THR+TYR+CYS\n")
        f.write("color marine, resn ARG+LYS+HIS+ASP+GLU\n")
        f.write(f"png {results[2]['path']}, width=800, height=600, ray=1\n")
        f.write("quit\n")

    pymol_cmd = ["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path]
    subprocess.run(pymol_cmd, check=True, capture_output=True)
    return results


def generate_pymol_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_pdb_path = _download_pdb_to_local(sequence, remote_pdb_path, file_prefix)

    img_filename = f"pymol_render_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    script_path = os.path.join(current_dir, f"render_{file_prefix}.pml")

    with open(script_path, 'w') as f:
        f.write(f"load {local_pdb_path}, myprotein\n")
        f.write("hide all\nshow sticks, myprotein\n")
        f.write("set stick_radius, 0.25\nset stick_ball, on\nset stick_ball_ratio, 1.5\n")
        f.write("color cpk, myprotein\ncolor gray70, elem c\n")
        f.write("bg_color black\nset ray_shadows, 0\nutil.cbc myprotein\nzoom\n")
        f.write(f"ray 800, 600\npng {img_path}\nquit\n")

    pymol_cmd = ["xvfb-run", "-a", "python", "-m", "pymol", "-c", "-q", script_path]
    try:
        print(f"⏳ [系統] 正在使用 PyMOL 渲染黑底棒狀模型...")
        subprocess.run(pymol_cmd, check=True, capture_output=True, text=True)
        if os.path.exists(script_path): os.remove(script_path)
        if os.path.exists(local_pdb_path): os.remove(local_pdb_path)
        print(f"✅ 圖片已生成: {img_path}")
        return img_path
    except subprocess.CalledProcessError as e:
        error_info = e.stderr if e.stderr else e.stdout
        raise Exception(f"PyMOL 渲染失敗。詳細資訊:\n{error_info}")


def generate_local_3d_plot(sequence: str, remote_pdb_path: str, file_prefix: str = "3D") -> str:
    pdb_data = ""
    if os.path.exists(remote_pdb_path):
        try:
            with open(remote_pdb_path, 'r') as f:
                pdb_data = f.read()
        except Exception:
            pass

    if not pdb_data:
        response = requests.post("https://api.esmatlas.com/foldSequence/v1/pdb/", data=sequence, timeout=30)
        pdb_data = response.text

    x, y, z = [], [], []
    for line in pdb_data.split('\n'):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                x.append(float(line[30:38]))
                y.append(float(line[38:46]))
                z.append(float(line[46:54]))
            except ValueError:
                pass

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='mediumpurple', linewidth=2.5)
    ax.scatter(x, y, z, color='darkslateblue', s=15)
    ax.set_title(f"Protein 3D Backbone Trace", fontsize=14)
    ax.axis('off')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_filename = f"local_3d_plot_{file_prefix}_{datetime.now().strftime('%H%M%S')}.png"
    img_path = os.path.join(current_dir, img_filename)
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    return img_path


# ==========================================
# 📄 PDF 報告工具
# ==========================================
@tool
def generate_pdf_report_tool(query: str = "生成報告") -> str:
    """彙整本次對話中所有工具的執行紀錄，生成最終 PDF 報告。"""
    try:
        pdf = FPDF()
        pdf.add_page()
        font_path = "/home/tempadmin/agentic_ai/msjh.ttc"

        has_unicode_font = False
        if os.path.exists(font_path):
            pdf.add_font("msjh", style="", fname=font_path)
            pdf.set_font("msjh", size=16)
            has_unicode_font = True
        else:
            pdf.set_font("helvetica", size=16)

        pdf.cell(200, 10, text="Protein Research Comprehensive Report",
                 new_x="LMARGIN", new_y="NEXT", align='C')

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

            lines = safe_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    pdf.ln(2)
                    continue
                line = re.sub(r'-{15,}', '--------------------', line)
                pdf.set_x(pdf.l_margin)
                try:
                    pdf.multi_cell(0, 8, text=line)
                except Exception as cell_e:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 8, text=line[:60] + " ... (文字過長截斷保護)")

                png_match = re.search(r'([\w\./_-]+\.png)', line)
                if png_match:
                    img_path = png_match.group(1)
                    if os.path.exists(img_path):
                        pdf.ln(2)
                        try:
                            pdf.image(img_path, w=150)
                            pdf.ln(5)
                        except Exception as img_e:
                            pdf.multi_cell(0, 8, text=f"[圖片載入失敗: {str(img_e)}]")

        report_name = f"Full_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(report_name)
        return f"✅ 完整報告已生成（包含分析圖表）：{report_name}"
    except Exception as e:
        return f"❌ PDF 生成失敗: {str(e)}"


# ==========================================
# 🤖 LLM 設定
# ==========================================
llm = ChatOllama(
    model="gemma2:9b",
    temperature=0,
    base_url=f"{BASE_TUNNEL_URL}/ollama"
)


# ==========================================
# 🛠️ 各項 AI 工具（原樣保留）
# ==========================================

@tool
def full_auto_analysis_tool(query: str) -> str:
    """一鍵執行全自動化分析。輸入格式為 '序列,數量'。"""
    parts = str(query).split(',')
    target_seq = clean_sequence(parts[0])
    num_variants = int(re.search(r'\d+', parts[1]).group()) if len(parts) > 1 else 10
    url = f"{BASE_TUNNEL_URL}/auto_research_pipeline"
    try:
        start_time = time.time()
        start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        response = requests.post(url, json={"sequence": target_seq, "num_variants": num_variants}, timeout=600)
        data = response.json()
        end_time = time.time()
        end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = end_time - start_time

        if data["status"] == "success":
            best_seq = data['best_variant']
            local_img_path = generate_local_plot(best_seq, "BestVariant")
            res_summary = (f"原始序列: {data['original_seq']}\n"
                           f"最優變體: {best_seq}\n"
                           f"MW: {data['physicochemical']['molecular_weight']}\n"
                           f"pI: {data['physicochemical']['isoelectric_point']}\n"
                           f"本機繪圖路徑: {local_img_path}\n"
                           f"PDB路徑: {data['pdb_path']}\n"
                           f"⏱️ 變體生成開始時間: {start_dt}\n"
                           f"⏱️ 變體生成結束時間: {end_dt}\n"
                           f"⏱️ 變體生成總耗時: {format_duration(elapsed)}")
            add_log("Full Auto Analysis", res_summary)
            return f"[OK] 全自動分析完成！\n{res_summary}"
        return f"[ERROR] 分析失敗：{data.get('message')}"
    except Exception as e:
        return f"異常: {str(e)}"


@tool
def neuropeptide_predictor_tool(sequence: str) -> str:
    """用於分析單一蛋白質序列是否為神經肽。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/predict"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "error":
            return f"伺服器報錯：{data.get('message')}"
        res = data.get("prediction", {})
        result_text = (f"序列: {clean_seq}\n預測結果: 【{res.get('label')}】\n"
                       f"預測機率: 【{res.get('probability', '0.0')}】\n資料來源: 【{res.get('note')}】")
        add_log("Neuropeptide Prediction", result_text)
        return f"預測完畢！\n{result_text}"
    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"


@tool
def anti_aging_predictor_tool(sequence: str) -> str:
    """用於分析單一蛋白質序列是否具有抗老化(Anti-Aging)活性。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/anti_aging_predictor"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "error":
            return f"伺服器報錯：{data.get('message')}"
        res = data.get("prediction", {})
        result_text = (f"序列: {clean_seq}\n預測結果: 【{res.get('label')}】\n"
                       f"預測機率: 【{res.get('probability', '0.0')}】\n資料來源: 【{res.get('note')}】")
        add_log("Anti-Aging Prediction", result_text)
        return f"🧬 抗老化預測完畢！\n{result_text}"
    except Exception as e:
        return f"❌ 抗老化工具呼叫失敗: {str(e)}"


@tool
def hemolytic_predictor_tool(sequence: str) -> str:
    """用於分析單一蛋白質序列是否具有溶血性。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/hemo_predictor"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=180)
        data = response.json()
        if data.get("status") == "error":
            return f"伺服器報錯：{data.get('message')}"
        res = data.get("prediction", {})
        result_text = (f"序列: {clean_seq}\n預測結果: 【{res.get('label')}】\n"
                       f"預測機率: 【{res.get('probability', '0.0')}】\n資料來源: 【{res.get('note')}】")
        add_log("Hemolytic Prediction", result_text)
        return f"🩸 溶血性預測完畢！\n{result_text}"
    except Exception as e:
        return f"❌ 溶血性工具呼叫失敗: {str(e)}"


@tool
def ankh_protein_language_tool(sequence: str) -> str:
    """當使用者詢問蛋白質的深層特徵、Embedding 或語言模型分析時使用。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/ankh"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=3000)
        data = response.json()
        if data.get("status") == "success":
            result_text = f"序列: {clean_seq}\nAnkh 特徵摘要：{data.get('result', '無摘要')}"
            add_log("Ankh Protein Language Tool", result_text)
            return f"✅ Ankh 分析完成！\n{result_text}"
        error_msg = data.get('message', '未知錯誤')
        return f"❌ Ankh 工具報錯：{error_msg}"
    except Exception as e:
        return f"❌ 呼叫 Ankh 失敗: {str(e)}"


@tool
def protein_discovery_pipeline_tool(query: str) -> str:
    """執行蛋白質發現與進化管線，並對每條變體進行三重預測。傳入格式：序列,數量"""
    clean_query = str(query).replace("'", "").replace('"', '').replace('{', '').replace('}', '').strip()
    parts = clean_query.split(',')
    clean_seq = clean_sequence(parts[0])

    if not clean_seq or len(clean_seq) < 2:
        return "❌ 錯誤：請提供正確的蛋白質序列。"

    num_variants = 10
    if len(parts) > 1:
        try:
            num_match = re.search(r'\d+', parts[1])
            if num_match:
                num_variants = int(num_match.group())
        except Exception:
            num_variants = 10

    url = f"{BASE_TUNNEL_URL}/discovery_pipeline"
    try:
        start_time = time.time()
        start_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        response = requests.post(url, json={"sequence": clean_seq, "num_variants": num_variants}, timeout=300)
        data = response.json()
        gen_end = time.time()
        gen_end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        gen_elapsed = gen_end - start_time

        if data.get("status") != "success":
            return f"❌ 伺服器回傳錯誤: {data.get('message')}"

        results = data.get("top_results", [])
        if not results:
            return "✅ 管線執行完畢，但未能找到任何候選序列。"

        pred_start = time.time()
        enriched_results = []
        for i, r in enumerate(results, 1):
            seq = r['seq']
            record = {
                "seq": seq,
                "neuropeptide_label": r.get('label', 'N/A'),
                "neuropeptide_score": float(r.get('score', r.get('probability', 0))),
                "anti_aging_label": "N/A", "anti_aging_score": 0.0,
                "hemo_label": "N/A", "hemo_score": 0.0,
            }
            try:
                ar = requests.post(f"{BASE_TUNNEL_URL}/anti_aging_predictor",
                                   json={"sequence": seq}, timeout=120).json()
                if ar.get("status") == "success":
                    p = ar["prediction"]
                    record["anti_aging_label"] = p.get("label", "N/A")
                    record["anti_aging_score"] = float(p.get("probability", 0))
            except Exception:
                pass
            try:
                hr = requests.post(f"{BASE_TUNNEL_URL}/hemo_predictor",
                                   json={"sequence": seq}, timeout=180).json()
                if hr.get("status") == "success":
                    p = hr["prediction"]
                    record["hemo_label"] = p.get("label", "N/A")
                    record["hemo_score"] = float(p.get("probability", 0))
            except Exception:
                pass

            record["composite_score"] = round(
                record["neuropeptide_score"] + record["anti_aging_score"] - record["hemo_score"], 4
            )
            enriched_results.append(record)

        enriched_results.sort(key=lambda x: x["composite_score"], reverse=True)

        pred_end = time.time()
        pred_elapsed = pred_end - pred_start
        total_elapsed = pred_end - start_time
        end_dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        time_info = (f"⏱️ 變體生成開始: {start_dt}\n"
                     f"⏱️ 變體生成結束: {gen_end_dt}（耗時 {format_duration(gen_elapsed)}）\n"
                     f"⏱️ 三重預測耗時: {format_duration(pred_elapsed)}\n"
                     f"⏱️ 全流程結束: {end_dt}（總耗時 {format_duration(total_elapsed)}）\n")

        res_text = (f"✅ 管線執行成功（含三重預測）！\n"
                    f"- 生成變體要求: {data.get('total_generated', num_variants)} 條\n"
                    f"- 去冗餘後剩餘: {data.get('after_cdhit', len(results))} 條\n"
                    f"{time_info}\n📊 候選序列三重活性預測：\n")

        res_text += ("\n| 排名 | 序列 | 神經肽 | 機率 | 抗老化 | 機率 | 溶血性 | 機率 | 綜合分數 |\n"
                     "|------|------|--------|------|--------|------|--------|------|----------|\n")
        for i, r in enumerate(enriched_results, 1):
            res_text += (f"| {i} | {r['seq']} | {r['neuropeptide_label']} | {r['neuropeptide_score']:.4f} | "
                         f"{r['anti_aging_label']} | {r['anti_aging_score']:.4f} | "
                         f"{r['hemo_label']} | {r['hemo_score']:.4f} | {r['composite_score']} |\n")

        res_text += "\n💡 綜合分數 = 神經肽機率 + 抗老化機率 − 溶血性機率\n"
        add_log("Protein Discovery Pipeline (Triple Prediction)", res_text)
        return res_text
    except Exception as e:
        return f"❌ 連線失敗: {str(e)}"


@tool
def secondary_structure_tool(sequence: str) -> str:
    """分析二級結構 (Helix/Sheet/Coil)。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/secondary_structure"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=60)
        data = response.json()
        if data.get("status") == "success":
            result_text = (f"二級結構分析完成！\n序列: {data['sequence']}\n"
                           f"結構: {data['structure']}\n統計: {data['summary']}")
            add_log("Secondary Structure Analysis", result_text)
            return result_text
        return f"伺服器報錯：{data.get('message')}"
    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"


@tool
def esmfold_3d_structure_tool(sequence: str) -> str:
    """預測立體結構，並使用 PyMOL 生成高畫質 3D 棒狀模型圖。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/esmfold"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=120)
        data = response.json()
        if data.get("status") == "success":
            remote_pdb_path = data['file_path']
            try:
                local_img_path = generate_pymol_plot(clean_seq, remote_pdb_path, "3DStruct")
                img_log_text = f"- 本機 3D 繪圖路徑: {local_img_path}"
            except Exception:
                try:
                    fallback_path = generate_local_3d_plot(clean_seq, remote_pdb_path, "3DStruct")
                    img_log_text = f"- 本機 3D 繪圖路徑(備援): {fallback_path}"
                except Exception as e2:
                    img_log_text = f"- 本機 3D 繪圖失敗: {str(e2)}"
            result_text = (f"✅ 3D 結構預測大成功！\n"
                           f"- 遠端 PDB 檔案儲存於：【{remote_pdb_path}】\n{img_log_text}")
            add_log("ESMFold 3D Structure Tool", result_text)
            return result_text
        return f"伺服器報錯：{data.get('message')}"
    except Exception as e:
        return f"❌ 工具呼叫失敗: {str(e)}"


def full_peptide_analysis(sequence: str) -> str:
    clean_seq = clean_sequence(sequence)
    if len(clean_seq) < 5:
        return f"❌ 序列太短（{len(clean_seq)}），需 ≥ 5 個氨基酸"

    base_url = BASE_TUNNEL_URL
    results = []
    detail_log = [f"序列: {clean_seq}"]

    try:
        r = requests.post(f"{base_url}/predict", json={"sequence": clean_seq}, timeout=180).json()
        if r.get("status") == "success":
            p = r["prediction"]
            line = f"🧠 神經肽：{p['label']}（機率 {p['probability']}）"
            results.append(line); detail_log.append(line)
        else:
            results.append(f"🧠 神經肽：預測失敗 - {r.get('message')}")
    except Exception as e:
        results.append(f"🧠 神經肽：錯誤 - {e}")

    try:
        r = requests.post(f"{base_url}/anti_aging_predictor", json={"sequence": clean_seq}, timeout=180).json()
        if r.get("status") == "success":
            p = r["prediction"]
            line = f"🌱 抗老化：{p['label']}（機率 {p['probability']}）"
            results.append(line); detail_log.append(line)
        else:
            results.append(f"🌱 抗老化：預測失敗 - {r.get('message')}")
    except Exception as e:
        results.append(f"🌱 抗老化：錯誤 - {e}")

    try:
        r = requests.post(f"{base_url}/hemo_predictor", json={"sequence": clean_seq}, timeout=300).json()
        if r.get("status") == "success":
            p = r["prediction"]
            line = f"🩸 溶血性：{p['label']}（機率 {p['probability']}）"
            results.append(line); detail_log.append(line)
        else:
            results.append(f"🩸 溶血性：預測失敗 - {r.get('message')}")
    except Exception as e:
        results.append(f"🩸 溶血性：錯誤 - {e}")

    final_text = f"序列 {clean_seq} 的完整生物活性分析：\n" + "\n".join(results)
    add_log("Full Peptide Analysis (3-in-1)", "\n".join(detail_log))
    return final_text


full_analysis_tool = Tool(
    name="full_peptide_analysis_tool",
    func=full_peptide_analysis,
    description="一次性完整分析一條胜肽序列的三種生物活性（神經肽、抗老化、溶血性）。"
)


def batch_anti_aging_hemo_screening(sequences_str: str) -> str:
    raw = re.split(r'[,\n\s]+', sequences_str)
    sequences = [clean_sequence(s) for s in raw if clean_sequence(s) and len(clean_sequence(s)) >= 5]

    if not sequences:
        return "❌ 沒有有效序列（長度需 ≥ 5）"

    results = []
    base_url = BASE_TUNNEL_URL

    for idx, seq in enumerate(sequences, 1):
        record = {"seq": seq, "length": len(seq)}
        try:
            r = requests.post(f"{base_url}/anti_aging_predictor", json={"sequence": seq}, timeout=180).json()
            if r.get("status") == "success":
                p = r["prediction"]
                record["anti_aging_label"] = p["label"]
                record["anti_aging_prob"] = float(p["probability"])
            else:
                record["anti_aging_label"] = "error"
                record["anti_aging_prob"] = 0.0
        except Exception:
            record["anti_aging_label"] = "error"
            record["anti_aging_prob"] = 0.0

        try:
            r = requests.post(f"{base_url}/hemo_predictor", json={"sequence": seq}, timeout=300).json()
            if r.get("status") == "success":
                p = r["prediction"]
                record["hemo_label"] = p["label"]
                record["hemo_prob"] = float(p["probability"])
            else:
                record["hemo_label"] = "error"
                record["hemo_prob"] = 1.0
        except Exception:
            record["hemo_label"] = "error"
            record["hemo_prob"] = 1.0

        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            pa = ProteinAnalysis(seq)
            record["mw"] = round(pa.molecular_weight(), 2)
            record["pi"] = round(pa.isoelectric_point(), 2)
            record["gravy"] = round(pa.gravy(), 3)
        except Exception:
            record["mw"] = record["pi"] = record["gravy"] = "N/A"

        record["composite_score"] = round(record["anti_aging_prob"] - record["hemo_prob"], 4)
        results.append(record)

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_n = min(10, len(results))
    top_results = results[:top_n]

    table = "\n| 排名 | 序列 | 長度 | 抗老化標籤 | 抗老化機率 | 溶血標籤 | 溶血機率 | MW (Da) | pI | GRAVY | 綜合分數 |\n"
    table += "|------|------|------|-----------|-----------|---------|---------|---------|-----|-------|----------|\n"
    for i, r in enumerate(top_results, 1):
        table += (f"| {i} | {r['seq']} | {r['length']} | {r['anti_aging_label']} | "
                  f"{r['anti_aging_prob']:.4f} | {r['hemo_label']} | {r['hemo_prob']:.4f} | "
                  f"{r['mw']} | {r['pi']} | {r['gravy']} | {r['composite_score']} |\n")

    summary = (f"✅ 批次篩選完成！共分析 {len(sequences)} 條序列，輸出 Top {top_n}\n"
               f"排序規則：綜合分數 = 抗老化機率 − 溶血機率\n{table}")
    add_log("Batch Anti-Aging + Hemo Screening", summary)
    return summary


batch_screening_tool = Tool(
    name="batch_anti_aging_hemo_screening_tool",
    func=batch_anti_aging_hemo_screening,
    description="批次篩選工具，當使用者提供多條序列並要求排序篩選時使用。"
)


@tool
def protein_physicochemical_plot_tool(sequence: str) -> str:
    """疏水性、等電點與分子量分析。"""
    clean_seq = clean_sequence(sequence)
    url = f"{BASE_TUNNEL_URL}/analyze_physicochemical"
    try:
        response = requests.post(url, json={"sequence": clean_seq}, timeout=30)
        data = response.json()
        if data["status"] == "success":
            local_img_path = generate_local_plot(clean_seq, "Physico")
            result_text = (f"📊 理化性質分析完成！\n"
                           f"- 分子量 (MW): {data['molecular_weight']}\n"
                           f"- 等電點 (pI): {data['isoelectric_point']}\n"
                           f"- 遠端原圖路徑: {data['image_path']}\n"
                           f"- 本機繪圖路徑: {local_img_path}")
            add_log("Protein Physicochemical Plot Tool", result_text)
            return result_text
        return f"錯誤：{data['message']}"
    except Exception as e:
        return f"呼叫工具失敗: {str(e)}"


# ==========================================
# 🤖 Agent 初始化
# ==========================================
tools = [
    batch_screening_tool, full_analysis_tool, full_auto_analysis_tool,
    neuropeptide_predictor_tool, anti_aging_predictor_tool, hemolytic_predictor_tool,
    ankh_protein_language_tool, protein_discovery_pipeline_tool,
    secondary_structure_tool, esmfold_3d_structure_tool,
    protein_physicochemical_plot_tool, generate_pdf_report_tool
]

agent = initialize_agent(
    tools, llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, max_iterations=2, handle_parsing_errors=True,
    agent_kwargs={
        "prefix": """你是一個自動化生物研究機器人。
        作業流程：
        1. 接收序列後，根據使用者需求選擇對應工具：
            - 「批次篩選 / 多條序列排序」→ batch_anti_aging_hemo_screening_tool
            - 「同時 / 完整分析 / 三種活性」→ full_peptide_analysis_tool
            - 「神經肽」→ neuropeptide_predictor_tool
            - 「抗老化」→ anti_aging_predictor_tool
            - 「溶血」→ hemolytic_predictor_tool
            - 「全面分析」→ full_auto_analysis_tool
            - 「3D 結構」→ esmfold_3d_structure_tool
            - 「理化性質」→ protein_physicochemical_plot_tool
            - 「二級結構」→ secondary_structure_tool
            - 「Embedding」→ ankh_protein_language_tool
            - 「演化 / 變體生成」→ protein_discovery_pipeline_tool
        2. 完成後主動呼叫 generate_pdf_report_tool 生成 PDF 報告。"""
    }
)


# ==========================================
# 🖥️ ChatGPT 風格 GUI
# ==========================================
class StdoutRedirector:
    """將 stdout 導向 GUI 的 queue"""
    def __init__(self, queue_obj):
        self.queue = queue_obj
        self.terminal = sys.__stdout__

    def write(self, msg):
        self.terminal.write(msg)
        if msg.strip():
            self.queue.put(("log", msg))

    def flush(self):
        self.terminal.flush()


from tkinter import font as tkfont
from PIL import Image, ImageTk
import queue
import threading
import sys
import os
import re

# ⭐ 全域外觀設定（放在檔案最上方，import 之後）
ctk.set_appearance_mode("dark")        # "dark" / "light" / "system"
ctk.set_default_color_theme("green")   # "blue" / "green" / "dark-blue"


class ChatGPTStyleGUI:
    # ChatGPT 風格配色
    BG_MAIN      = "#343541"
    BG_SIDEBAR   = "#202123"
    BG_USER_MSG  = "#343541"
    BG_BOT_MSG   = "#444654"
    BG_INPUT     = "#40414F"
    FG_TEXT      = "#ECECF1"
    FG_SECONDARY = "#8E8EA0"
    ACCENT       = "#10A37F"
    ACCENT_HOVER = "#0D8C6D"
    BORDER       = "#565869"
    HOVER_DARK   = "#2A2B32"

    def __init__(self, root):
        self.root = root
        self.root.title("Protein Research Agent")
        self.root.geometry("1280x860")
        self.root.configure(fg_color=self.BG_MAIN)
        self.root.minsize(900, 650)

        # === 字型（customtkinter 用 CTkFont） ===
        # ⭐ customtkinter 會自動處理中文字型，直接指定即可
        self.font_main   = ctk.CTkFont(family="Noto Sans CJK TC", size=14)
        self.font_bold   = ctk.CTkFont(family="Noto Sans CJK TC", size=14, weight="bold")
        self.font_title  = ctk.CTkFont(family="Noto Sans CJK TC", size=18, weight="bold")
        self.font_small  = ctk.CTkFont(family="Noto Sans CJK TC", size=12)
        self.font_symbol = ctk.CTkFont(family="Noto Sans CJK TC", size=20, weight="bold")
        self.font_btn    = ctk.CTkFont(family="Noto Sans CJK TC", size=18, weight="bold")
        # tk.Text 用的（不能用 CTkFont）
        self.font_text   = tkfont.Font(family="Noto Sans CJK TC", size=14)

        # Queue 跨執行緒通訊
        self.msg_queue = queue.Queue()

        # 重定向 stdout
        sys.stdout = StdoutRedirector(self.msg_queue)

        self._build_ui()
        self._poll_queue()

        # 歡迎訊息
        self.append_bot_message(
            "您好，我是 Protein Research Agent ！\n"
            "\n"
            "我可以協助您進行以下分析：\n"
            "\n"
            "    ◆   神經肽預測\n"
            "    ◆   抗老化活性預測\n"
            "    ◆   溶血性預測\n"
            "    ◆   3D 結構預測 (ESMFold)\n"
            "    ◆   理化性質分析\n"
            "    ◆   變體生成與演化管線\n"
            "    ◆   自動生成 PDF 報告\n"
            "\n"
            "請在下方輸入您的指令，例如：\n"
            "「分析序列 GPRLVRF 的所有活性」"
        )

    def _build_ui(self):
        # === 主容器 ===
        main_container = ctk.CTkFrame(self.root, fg_color=self.BG_MAIN, corner_radius=0)
        main_container.pack(fill="both", expand=True)

        # === 左側欄 ===
        sidebar = ctk.CTkFrame(main_container, fg_color=self.BG_SIDEBAR,
                                width=280, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # 標題區
        title_frame = ctk.CTkFrame(sidebar, fg_color=self.BG_SIDEBAR, corner_radius=0)
        title_frame.pack(fill="x", padx=18, pady=20)
        ctk.CTkLabel(title_frame, text="◆  Protein Agent",
                     font=self.font_title, text_color=self.FG_TEXT,
                     anchor="w").pack(fill="x")
        ctk.CTkLabel(title_frame, text="AI 蛋白質研究助理",
                     font=self.font_small, text_color=self.FG_SECONDARY,
                     anchor="w").pack(fill="x", pady=(6, 0))

        # 新對話按鈕
        new_chat_btn = ctk.CTkButton(sidebar, text="＋   新對話",
                                      font=self.font_main,
                                      fg_color=self.BG_SIDEBAR,
                                      hover_color=self.HOVER_DARK,
                                      text_color=self.FG_TEXT,
                                      border_color=self.BORDER,
                                      border_width=1,
                                      height=42,
                                      anchor="w",
                                      command=self.clear_chat)
        new_chat_btn.pack(fill="x", padx=18, pady=6)

        # PDF 按鈕
        pdf_btn = ctk.CTkButton(sidebar, text="▤   生成 PDF 報告",
                                 font=self.font_main,
                                 fg_color=self.BG_SIDEBAR,
                                 hover_color=self.HOVER_DARK,
                                 text_color=self.FG_TEXT,
                                 border_color=self.BORDER,
                                 border_width=1,
                                 height=42,
                                 anchor="w",
                                 command=self.generate_pdf)
        pdf_btn.pack(fill="x", padx=18, pady=6)

        # 分隔線
        ctk.CTkFrame(sidebar, fg_color=self.BORDER, height=1, corner_radius=0)\
            .pack(fill="x", padx=18, pady=15)

        # 範例指令標題
        ctk.CTkLabel(sidebar, text="◇   範例指令", font=self.font_bold,
                     text_color=self.FG_SECONDARY, anchor="w")\
            .pack(fill="x", padx=18, pady=(8, 8))

        examples = [
            "分析 GPRLVRF 完整活性",
            "預測 KLAKLAKKLAKLAK 溶血性",
            "對 GPRLVRF 生成 5 條變體",
            "分析 GPRLVRF 的 3D 結構",
        ]
        for ex in examples:
            btn = ctk.CTkButton(sidebar, text="  ▸  " + ex,
                                font=self.font_small,
                                fg_color=self.BG_SIDEBAR,
                                hover_color=self.HOVER_DARK,
                                text_color=self.FG_SECONDARY,
                                border_width=0, height=34,
                                anchor="w",
                                command=lambda t=ex: self.set_input(t))
            btn.pack(fill="x", padx=18, pady=3)

        # 底部資訊
        bottom_frame = ctk.CTkFrame(sidebar, fg_color=self.BG_SIDEBAR, corner_radius=0)
        bottom_frame.pack(side="bottom", fill="x", padx=18, pady=18)
        ctk.CTkFrame(bottom_frame, fg_color=self.BORDER, height=1, corner_radius=0)\
            .pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(bottom_frame, text="◉   黃瑞麟",
                     font=self.font_main, text_color=self.FG_TEXT,
                     anchor="w").pack(fill="x")
        ctk.CTkLabel(bottom_frame, text="Powered by Gemma2:9b",
                     font=self.font_small, text_color=self.FG_SECONDARY,
                     anchor="w").pack(fill="x", pady=(6, 0))

        # === 右側聊天區 ===
        chat_area = ctk.CTkFrame(main_container, fg_color=self.BG_MAIN, corner_radius=0)
        chat_area.pack(side="left", fill="both", expand=True)

        # ⭐ 訊息顯示區改用 CTkScrollableFrame（自動處理捲動）
        self.msg_frame = ctk.CTkScrollableFrame(chat_area,
                                                  fg_color=self.BG_MAIN,
                                                  corner_radius=0,
                                                  scrollbar_button_color=self.BORDER,
                                                  scrollbar_button_hover_color=self.FG_SECONDARY)
        self.msg_frame.pack(fill="both", expand=True)

        # === 底部輸入區 ===
        input_outer = ctk.CTkFrame(chat_area, fg_color=self.BG_MAIN, corner_radius=0)
        input_outer.pack(fill="x", padx=24, pady=(0, 24))

        input_container = ctk.CTkFrame(input_outer,
                                         fg_color=self.BG_INPUT,
                                         border_color=self.BORDER,
                                         border_width=1,
                                         corner_radius=8)
        input_container.pack(fill="x")

        # ⭐ 輸入框：CTkTextbox 對中文支援更好
        self.input_text = ctk.CTkTextbox(input_container, height=80,
                                           font=self.font_main,
                                           fg_color=self.BG_INPUT,
                                           text_color=self.FG_TEXT,
                                           border_width=0,
                                           wrap="word")
        self.input_text.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        self.input_text.bind("<Return>", self._on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None)

        # 送出按鈕
        self.send_btn = ctk.CTkButton(input_container, text="→",
                                       font=self.font_btn,
                                       fg_color=self.ACCENT,
                                       hover_color=self.ACCENT_HOVER,
                                       text_color="white",
                                       width=60, height=60,
                                       corner_radius=6,
                                       command=self.send_message)
        self.send_btn.pack(side="right", padx=10, pady=10)

        # 狀態列
        self.status_label = ctk.CTkLabel(input_outer,
                                           text="按 Enter 傳送   |   Shift + Enter 換行",
                                           font=self.font_small,
                                           text_color=self.FG_SECONDARY)
        self.status_label.pack(pady=(12, 0))

    # ============= 事件處理 =============
    def _on_enter(self, event):
        if event.state & 0x0001:  # Shift
            return None
        self.send_message()
        return "break"

    def set_input(self, text):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", text)
        self.input_text.focus()

    def clear_chat(self):
        for w in self.msg_frame.winfo_children():
            w.destroy()
        self.append_bot_message("新的對話已開始，請輸入您的問題！")

    # ============= 訊息渲染 =============
    def append_user_message(self, text):
        row = ctk.CTkFrame(self.msg_frame, fg_color=self.BG_USER_MSG, corner_radius=0)
        row.pack(fill="x")

        inner = ctk.CTkFrame(row, fg_color=self.BG_USER_MSG, corner_radius=0)
        inner.pack(fill="x", padx=60, pady=24)

        # 頭像
        avatar = ctk.CTkLabel(inner, text="◉", font=self.font_symbol,
                                fg_color="#5436DA", text_color="white",
                                width=40, height=40, corner_radius=4)
        avatar.pack(side="left", anchor="n", padx=(0, 18))

        # 內容
        content_frame = ctk.CTkFrame(inner, fg_color=self.BG_USER_MSG, corner_radius=0)
        content_frame.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(content_frame, text="您",
                     font=self.font_bold, text_color=self.FG_TEXT,
                     anchor="w").pack(fill="x")

        # ⭐ 訊息內文：用 tk.Text 才能多行顯示 + 控制行距
        line_count = text.count('\n') + max(1, len(text) // 70)
        height = min(max(line_count, 1), 20)

        msg_text = ctk.CTkTextbox(content_frame,
                            font=self.font_main,
                            fg_color=self.BG_BOT_MSG,
                            text_color=self.FG_TEXT,
                            border_width=0,
                            wrap="word",
                            height=height * 24,   # ⭐ CTk 用像素
                            activate_scrollbars=False)
        msg_text.insert("1.0", text)
        msg_text.configure(state="disabled")
        msg_text.pack(anchor="w", pady=(8, 0), fill="x")



        self._scroll_to_bottom()

    def append_bot_message(self, text, is_log=False):
        row = ctk.CTkFrame(self.msg_frame, fg_color=self.BG_BOT_MSG, corner_radius=0)
        row.pack(fill="x")

        inner = ctk.CTkFrame(row, fg_color=self.BG_BOT_MSG, corner_radius=0)
        inner.pack(fill="x", padx=60, pady=24)

        # 頭像
        bg_avatar = self.ACCENT if not is_log else self.FG_SECONDARY
        symbol = "◆" if not is_log else "⚙"
        avatar = ctk.CTkLabel(inner, text=symbol, font=self.font_symbol,
                                fg_color=bg_avatar, text_color="white",
                                width=40, height=40, corner_radius=4)
        avatar.pack(side="left", anchor="n", padx=(0, 18))

        # 內容
        content_frame = ctk.CTkFrame(inner, fg_color=self.BG_BOT_MSG, corner_radius=0)
        content_frame.pack(side="left", fill="x", expand=True)

        name = "Protein Agent" if not is_log else "系統日誌"
        ctk.CTkLabel(content_frame, text=name,
                     font=self.font_bold, text_color=self.FG_TEXT,
                     anchor="w").pack(fill="x")

        # 文字內容
        line_count = text.count('\n') + max(1, len(text) // 70)
        height = min(max(line_count + 2, 3), 35)

        msg_text = tk.Text(content_frame, font=self.font_text,
                           bg=self.BG_BOT_MSG, fg=self.FG_TEXT,
                           relief="flat", bd=0, wrap="word",
                           height=height, padx=0, pady=0,
                           spacing1=4, spacing2=8, spacing3=4,
                           highlightthickness=0)
        msg_text.insert("1.0", text)
        msg_text.configure(state="disabled")
        msg_text.pack(anchor="w", pady=(8, 0), fill="x")

        # 圖片插入
        png_matches = re.findall(r'([\w\./\\_-]+\.png)', text)
        for img_path in png_matches:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img.thumbnail((520, 420))
                    photo = ImageTk.PhotoImage(img)
                    img_label = tk.Label(content_frame, image=photo,
                                          bg=self.BG_BOT_MSG)
                    img_label.image = photo
                    img_label.pack(anchor="w", pady=(14, 0))
                except Exception as e:
                    print(f"圖片載入失敗: {e}")

        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        self.root.update_idletasks()
        # CTkScrollableFrame 的捲動到底
        self.msg_frame._parent_canvas.yview_moveto(1.0)

    # ============= 訊息傳送 =============
    def send_message(self):
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            return
        self.input_text.delete("1.0", "end")

        self.append_user_message(text)
        self.send_btn.configure(state="disabled", text="…")
        self.status_label.configure(text="AI 正在處理中，請稍候...")

        thread = threading.Thread(target=self._run_agent, args=(text,), daemon=True)
        thread.start()

    def _run_agent(self, user_input):
        try:
            result = agent.invoke({"input": user_input})
            output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            self.msg_queue.put(("bot", output))
        except Exception as e:
            self.msg_queue.put(("bot", f"[錯誤] 執行失敗：{str(e)}"))
        finally:
            self.msg_queue.put(("done", None))

    def generate_pdf(self):
        self.status_label.configure(text="正在生成 PDF 報告...")
        self.send_btn.configure(state="disabled")

        def _gen():
            try:
                result = generate_pdf_report_tool.invoke("")
                self.msg_queue.put(("bot", result))
            except Exception as e:
                self.msg_queue.put(("bot", f"[錯誤] PDF 生成失敗：{str(e)}"))
            finally:
                self.msg_queue.put(("done", None))

        threading.Thread(target=_gen, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                msg_type, content = self.msg_queue.get_nowait()
                if msg_type == "bot":
                    self.append_bot_message(content)
                elif msg_type == "log":
                    pass
                elif msg_type == "done":
                    self.send_btn.configure(state="normal", text="→")
                    self.status_label.configure(text="按 Enter 傳送   |   Shift + Enter 換行")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

# ============= 啟動 =============
if __name__ == "__main__":
    # ⭐ 注意：root 改用 ctk.CTk() 而不是 tk.Tk()
    root = ctk.CTk()
    app = ChatGPTStyleGUI(root)
    root.mainloop()