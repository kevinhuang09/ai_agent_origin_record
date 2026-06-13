import sys
import os
import csv
import subprocess
import requests
import random
import torch
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd

# ==========================================
# ⚙️ 系統與路徑設定
# ==========================================
OLD_ENV_PYTHON = "/home/g503/miniconda3/envs/neuropredictor-tool/bin/python"

# server.py 所在資料夾（預期為 /home/g503/ai_agent0520）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🟢 neuro_predictor 子專案根目錄
NEURO_ROOT = os.path.join(BASE_DIR, "neuro_predictor")

# 🟢 主程式資料夾：neuro_predictor/mainProgram/
MAIN_PROGRAM_DIR = os.path.join(NEURO_ROOT, "mainProgram")

# 🟢 data 資料夾：neuro_predictor/data/
DATA_DIR = os.path.join(NEURO_ROOT, "data")

# 🟢 神經肽預測腳本：neuro_predictor/mainProgram/predict_web.py
PREDICT_SCRIPT = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")

# 🟢 神經肽預測輸出 CSV：neuro_predictor/data/mlScore/final_prediction_results.csv
ML_SCORE_DIR = os.path.join(DATA_DIR, "mlScore")
PREDICT_CSV = os.path.join(ML_SCORE_DIR, "final_prediction_results.csv")
os.makedirs(ML_SCORE_DIR, exist_ok=True)

# 🟢 神經肽輸入 fasta：對應 predict_web.py 內 "../data/test.fasta"
FASTA_PATH = os.path.join(DATA_DIR, "test.fasta")
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# 🌱 抗老化預測（獨立腳本，與 server.py 同層）
# ==========================================
# 路徑（使用 anti_aging_test.fasta 對齊腳本內預設）
ANTI_AGING_DIR    = os.path.join(BASE_DIR, "Anti-angiogenic")
ANTI_AGING_SCRIPT = os.path.join(ANTI_AGING_DIR, "anti_aging_predictor.py")
ANTI_AGING_FASTA  = os.path.join(ANTI_AGING_DIR, "anti_aging_test.fasta")   # ← 改名
ANTI_AGING_CSV    = os.path.join(ANTI_AGING_DIR, "anti_aging_prediction_result.csv")

# 🟢 圖片輸出資料夾（放在 server.py 同層）
IMAGE_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(IMAGE_DIR, exist_ok=True)

# 把 mainProgram 加進 sys.path，方便 import 子模組
if MAIN_PROGRAM_DIR not in sys.path:
    sys.path.append(MAIN_PROGRAM_DIR)

# ---- 啟動時印出路徑檢查 ----
print("=" * 60)
print("🔍 路徑檢查")
print(f"  BASE_DIR           = {BASE_DIR}")
print(f"  NEURO_ROOT         = {NEURO_ROOT}            exists={os.path.isdir(NEURO_ROOT)}")
print(f"  MAIN_PROGRAM_DIR   = {MAIN_PROGRAM_DIR}      exists={os.path.isdir(MAIN_PROGRAM_DIR)}")
print(f"  DATA_DIR           = {DATA_DIR}              exists={os.path.isdir(DATA_DIR)}")
print(f"  PREDICT_SCRIPT     = {PREDICT_SCRIPT}        exists={os.path.isfile(PREDICT_SCRIPT)}")
print(f"  FASTA_PATH         = {FASTA_PATH}")
print(f"  ML_SCORE_DIR       = {ML_SCORE_DIR}          exists={os.path.isdir(ML_SCORE_DIR)}")
print(f"  PREDICT_CSV        = {PREDICT_CSV}")
print("-" * 60)
print(f"  ANTI_AGING_SCRIPT  = {ANTI_AGING_SCRIPT}     exists={os.path.isfile(ANTI_AGING_SCRIPT)}")
print(f"  ANTI_AGING_FASTA   = {ANTI_AGING_FASTA}")
print(f"  ANTI_AGING_CSV     = {ANTI_AGING_CSV}")
print("-" * 60)
print(f"  OLD_ENV_PYTHON     = {OLD_ENV_PYTHON}        exists={os.path.isfile(OLD_ENV_PYTHON)}")
print("=" * 60)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

AA_TO_ID = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)}
AA_TO_ID["<PAD>"] = 0
ID_TO_LABEL = {0: "C", 1: "H", 2: "E"}


# ==========================================
# 🔧 共用工具函式
# ==========================================
def _safe_float(v, default=0.0):
    """安全轉成 float，失敗回傳預設值"""
    try:
        return float(v)
    except Exception:
        return default


def _get_seq_id(row: dict) -> str:
    """從 CSV row 取出第一欄 seq id"""
    return (row.get('')
            or row.get('seq_id')
            or row.get('id')
            or row.get('ID')
            or '').strip()


def _get_score(row: dict) -> float:
    """從 CSV row 取出分數"""
    raw = (row.get('Average_Score')
           or row.get('lightgbm_final')
           or row.get('probability')
           or row.get('score')
           or 0.0)
    return _safe_float(raw, 0.0)


def _get_label(row: dict) -> str:
    """從 CSV row 取出標籤"""
    return (row.get('Final_Label')
            or row.get('label')
            or 'unknown')


# ==========================================
# 🧠 神經肽預測核心
# ==========================================
def do_prediction(sequence: str) -> dict:
    try:
        if len(sequence) < 5:
            return {"label": "error", "probability": "0.0",
                    "note": f"序列長度需 ≥ 5（目前 {len(sequence)}）"}

        if not os.path.isfile(PREDICT_SCRIPT):
            return {"label": "error", "probability": "0.0",
                    "note": f"找不到預測腳本: {PREDICT_SCRIPT}"}

        if os.path.exists(PREDICT_CSV):
            os.remove(PREDICT_CSV)

        with open(FASTA_PATH, "w") as f:
            f.write(f">seq1\n{sequence}\n")

        cmd = f"{OLD_ENV_PYTHON} {PREDICT_SCRIPT}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR
        )

        if result.returncode != 0:
            return {"label": "error", "probability": "0.0",
                    "note": f"報錯: {result.stderr.strip()[-300:]}"}

        if not os.path.exists(PREDICT_CSV):
            stdout_tail = (result.stdout or "").strip()[-400:] or "(無 stdout)"
            stderr_tail = (result.stderr or "").strip()[-200:] or "(無 stderr)"
            return {"label": "error", "probability": "0.0",
                    "note": f"未產生 CSV | stdout: {stdout_tail} | stderr: {stderr_tail}"}

        with open(PREDICT_CSV, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "probability": "0.0", "note": "CSV 為空"}

            latest_row = reader[-1]
            label = _get_label(latest_row)
            final_score = _get_score(latest_row)

            return {
                "label": label,
                "probability": f"{final_score:.4f}",
                "note": "AI Predicted"
            }
    except Exception as e:
        return {"label": "error", "probability": "0.0", "note": str(e)}


# ==========================================
# 🌱 抗老化預測核心
# ==========================================
def do_anti_aging_prediction(sequence: str) -> dict:
    """
    呼叫 anti_aging_predictor.py（與 server.py 同層）。
    cwd = BASE_DIR，產出 anti_aging_prediction_result.csv 也在 BASE_DIR。
    """
    try:
        if len(sequence) < 5:
            return {"label": "error", "probability": "0.0",
                    "note": f"序列長度需 ≥ 5（目前 {len(sequence)}）"}

        if not os.path.isfile(ANTI_AGING_SCRIPT):
            return {"label": "error", "probability": "0.0",
                    "note": f"找不到抗老化腳本: {ANTI_AGING_SCRIPT}"}

        # 清掉舊 CSV
        if os.path.exists(ANTI_AGING_CSV):
            os.remove(ANTI_AGING_CSV)

        # 寫入抗老化專用 fasta（在 BASE_DIR 同層）
        with open(ANTI_AGING_FASTA, "w") as f:
            f.write(f">seq1\n{sequence}\n")

        # 在 BASE_DIR 下執行（cwd 與 server.py 同層）
        cmd = f"{OLD_ENV_PYTHON} {ANTI_AGING_SCRIPT}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=BASE_DIR
        )

        if result.returncode != 0:
            return {"label": "error", "probability": "0.0",
                    "note": f"腳本錯誤: {result.stderr.strip()[-300:]}"}

        if not os.path.exists(ANTI_AGING_CSV):
            stdout_tail = (result.stdout or "").strip()[-400:] or "(無 stdout)"
            stderr_tail = (result.stderr or "").strip()[-200:] or "(無 stderr)"
            return {"label": "error", "probability": "0.0",
                    "note": f"未產生 CSV | stdout: {stdout_tail} | stderr: {stderr_tail}"}

        with open(ANTI_AGING_CSV, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "probability": "0.0", "note": "CSV 為空"}

            latest_row = reader[-1]
            label = _get_label(latest_row)
            final_score = _get_score(latest_row)

            return {
                "label": label,
                "probability": f"{final_score:.4f}",
                "note": "AI Predicted (Anti-Aging)"
            }
    except Exception as e:
        return {"label": "error", "probability": "0.0", "note": str(e)}


# ==========================================
# 📡 API 節點
# ==========================================
@app.post("/predict")
async def predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "No sequence"}
    res = do_prediction(sequence)
    if res["label"] == "error":
        return {"status": "error", "message": res["note"]}
    return {"status": "success", "prediction": res}


@app.post("/anti_aging_predictor")
async def anti_aging_predictor(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "No sequence"}
    res = do_anti_aging_prediction(sequence)
    if res["label"] == "error":
        return {"status": "error", "message": res["note"]}
    return {"status": "success", "prediction": res}


try:
    import ankh_service
    get_ankh_embedding = ankh_service.get_ankh_embedding
except Exception:
    def get_ankh_embedding(seq): raise Exception("Ankh 未載入")


@app.post("/ankh")
@app.post("/ankh_feature")
async def ankh_feature(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    try:
        embedding = get_ankh_embedding(sequence)
        summary = f"維度: {len(embedding)}, 前5數值: {[round(x, 4) for x in embedding[:5]]}"
        return {"status": "success", "result": summary, "full_dim": len(embedding)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/discovery_pipeline")
async def discovery_pipeline(data: dict):
    base_sequence = data.get("sequence", "").strip().upper()
    if not base_sequence:
        return {"status": "error", "message": "空序列"}
    if len(base_sequence) < 5:
        return {"status": "error", "message": f"序列長度需 ≥ 5（目前 {len(base_sequence)}）"}

    num_variants = data.get("num_variants", 100)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 100

    if not os.path.isfile(PREDICT_SCRIPT):
        return {"status": "error", "message": f"找不到預測腳本: {PREDICT_SCRIPT}"}

    input_fasta = os.path.join(MAIN_PROGRAM_DIR, "pipeline_in.fasta")
    output_cdhit = os.path.join(MAIN_PROGRAM_DIR, "pipeline_out")
    output_2d = os.path.join(MAIN_PROGRAM_DIR, "pipeline_2d_out")
    known_db = os.path.join(MAIN_PROGRAM_DIR, "cdhitusing.fasta")

    print(f"🚀 [Discovery] 從 {base_sequence} 衍生 {num_variants} 條序列...")

    variants = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(num_variants):
        seq_list = list(base_sequence)
        mutate_times = 1 if len(seq_list) <= 1 else random.randint(1, 2)
        for _ in range(mutate_times):
            pos = random.randint(0, len(seq_list) - 1)
            seq_list[pos] = random.choice(amino_acids)
        variants.append("".join(seq_list))

    with open(input_fasta, "w") as f:
        for i, v in enumerate(variants):
            f.write(f">var_{i}\n{v}\n")

    cmd_cdhit = f"cd-hit -i {input_fasta} -o {output_cdhit} -c 0.9 -n 5"
    subprocess.run(cmd_cdhit, shell=True, capture_output=True, text=True)

    final_output = output_cdhit
    if os.path.exists(known_db):
        print(f"🔍 [Discovery] 執行 CD-HIT-2D，比對已知資料庫：{known_db}")
        cmd_2d = f"cd-hit-2d -i {known_db} -i2 {output_cdhit} -o {output_2d} -c 0.9 -n 5"
        subprocess.run(cmd_2d, shell=True, capture_output=True, text=True)
        if os.path.exists(output_2d):
            final_output = output_2d
    else:
        print(f"⚠️ [Discovery] 找不到 {known_db}，跳過 2D 比對")

    final_candidates = []
    if os.path.exists(final_output):
        with open(final_output, "r") as f:
            current_id, current_seq = None, ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        final_candidates.append((current_id, current_seq))
                    current_id = line[1:].split()[0]
                    current_seq = ""
                else:
                    current_seq += line
            if current_id and current_seq:
                final_candidates.append((current_id, current_seq))

    if not final_candidates:
        print("⚠️ 篩選後無剩餘序列，啟用備用機制...")
        unique_seqs = list(set(variants))
        final_candidates = [(f"var_{i}", s) for i, s in enumerate(unique_seqs)]

    print(f"🔬 [Discovery] 最終剩餘 {len(final_candidates)} 條，開始 AI 預測...")

    if os.path.exists(PREDICT_CSV):
        os.remove(PREDICT_CSV)

    with open(FASTA_PATH, "w") as f:
        for sid, seq in final_candidates:
            f.write(f">{sid}\n{seq}\n")

    cmd_predict = f"{OLD_ENV_PYTHON} {PREDICT_SCRIPT}"
    subprocess.run(cmd_predict, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)

    seq_map = {sid: s for sid, s in final_candidates}
    results = []
    if os.path.exists(PREDICT_CSV):
        with open(PREDICT_CSV, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            for idx, row in enumerate(reader):
                seq_id = _get_seq_id(row)
                seq_text = seq_map.get(seq_id)
                if not seq_text and idx < len(final_candidates):
                    seq_id, seq_text = final_candidates[idx]
                if not seq_text:
                    continue
                results.append({
                    "seq_id": seq_id,
                    "seq": seq_text,
                    "label": _get_label(row),
                    "probability": _get_score(row)
                })

    temp_files = [
        input_fasta, output_cdhit, output_cdhit + ".clstr",
        output_2d, output_2d + ".clstr", FASTA_PATH
    ]
    for f_path in temp_files:
        if os.path.exists(f_path):
            os.remove(f_path)

    top_results = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]

    return {
        "status": "success",
        "total_generated": num_variants,
        "after_cdhit": len(final_candidates),
        "top_results": top_results
    }


@app.post("/esmfold")
async def esmfold_predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence or len(sequence) < 3:
        return {"status": "error", "message": "序列過短或無效"}
    try:
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        headers = {"Content-Type": "text/plain"}
        response = requests.post(url, data=sequence, headers=headers, timeout=60)
        if response.status_code == 200:
            file_name = f"esmfold_{sequence[:15]}.pdb"
            file_path = os.path.join(BASE_DIR, file_name)
            with open(file_path, "w") as f:
                f.write(response.text)
            return {"status": "success", "message": "3D 結構預測完成", "file_path": file_path}
        else:
            return {"status": "error", "message": f"ESMFold API 報錯，狀態碼: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze_physicochemical")
async def analyze_physicochemical(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "無效序列"}
    try:
        analysed_seq = ProteinAnalysis(sequence)
        hydrophobicity = analysed_seq.protein_scale(kd, window=5, edge=0.4)

        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(hydrophobicity) + 1), hydrophobicity, marker='o', linestyle='-', color='b')
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.title(f"Hydrophobicity Profile (Kyte-Doolittle) - {sequence[:10]}...")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophobicity Score")
        plt.grid(True, alpha=0.3)

        file_name = f"phys_plot_{sequence[:10]}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        plt.savefig(file_path)
        plt.close()

        mw = analysed_seq.molecular_weight()
        pi = analysed_seq.isoelectric_point()

        return {
            "status": "success",
            "image_path": file_path,
            "molecular_weight": f"{mw:.2f} Da",
            "isoelectric_point": f"{pi:.2f}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/secondary_structure")
async def secondary_structure(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "空序列"}
    try:
        pred_list = [random.choice([0, 1, 2]) for _ in sequence]
        structure = "".join([ID_TO_LABEL.get(pred_id, "C") for pred_id in pred_list])
        summary = (f"Alpha 螺旋(H): {structure.count('H')}, "
                   f"Beta 摺疊(E): {structure.count('E')}, "
                   f"無規捲曲(C): {structure.count('C')}")
        return {"status": "success", "sequence": sequence, "structure": structure, "summary": summary}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/auto_research_pipeline")
async def auto_research_pipeline(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    num_variants = data.get("num_variants", 10)
    if not sequence:
        return {"status": "error", "message": "未提供序列"}
    try:
        print(f"🔄 啟動一條龍分析管線，目標序列：{sequence}，數量：{num_variants}")
        discovery_res = await discovery_pipeline({
            "sequence": sequence,
            "num_variants": num_variants
        })
        if discovery_res.get("status") != "success" or not discovery_res.get("top_results"):
            return {"status": "error", "message": "變體生成失敗"}

        top_variant = discovery_res["top_results"][0]["seq"]
        variant_probability = discovery_res["top_results"][0].get("probability", 0.0)
        variant_label = discovery_res["top_results"][0].get("label", "unknown")

        phys_res = await analyze_physicochemical({"sequence": top_variant})
        esm_res = await esmfold_predict({"sequence": top_variant})

        return {
            "status": "success",
            "total_generated": discovery_res.get("total_generated"),
            "after_cdhit": discovery_res.get("after_cdhit"),
            "original_seq": sequence,
            "best_variant": top_variant,
            "variant_label": variant_label,
            "variant_probability": variant_probability,
            "physicochemical": phys_res if phys_res.get("status") == "success" else {
                "molecular_weight": "計算失敗",
                "isoelectric_point": "計算失敗",
                "image_path": "生成失敗"
            },
            "pdb_path": esm_res.get("file_path", "預測失敗")
        }
    except Exception as e:
        return {"status": "error", "message": f"Pipeline failed: {str(e)}"}


@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(path: str, request: Request):
    url = f"http://127.0.0.1:11434/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    try:
        response = requests.request(method=request.method, url=url, data=body, headers=headers, stream=True)
        return StreamingResponse(response.iter_content(chunk_size=1024), status_code=response.status_code)
    except Exception:
        return {"error": "連線失敗"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)
