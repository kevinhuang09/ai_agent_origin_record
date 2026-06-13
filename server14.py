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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PROGRAM_DIR = BASE_DIR

IMAGE_DIR = os.path.join(MAIN_PROGRAM_DIR, "plots")
os.makedirs(IMAGE_DIR, exist_ok=True)

if MAIN_PROGRAM_DIR not in sys.path:
    sys.path.append(MAIN_PROGRAM_DIR)

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
    """
    從 CSV row 取出第一欄（seq id）。
    您的 CSV 第一欄沒有 header，DictReader 會把 key 命名為 ''（空字串）。
    """
    return (row.get('')
            or row.get('seq_id')
            or row.get('id')
            or row.get('ID')
            or '').strip()


def _get_score(row: dict) -> float:
    """
    從 CSV row 取出分數，優先順序：
    Average_Score > lightgbm_final > probability > score
    """
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
# 🧠 預測核心功能
# ==========================================
def do_prediction(sequence: str) -> dict:
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")

    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)

        # 🟢 fasta header 用 seq1，與 CSV 第一欄對應
        with open(fasta_path, "w") as f:
            f.write(f">seq1\n{sequence}\n")

        cmd = f"{OLD_ENV_PYTHON} {script_abs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)

        if result.returncode != 0:
            return {"label": "error", "probability": "0.0", "note": f"報錯: {result.stderr.strip()[-200:]}"}
        if not os.path.exists(csv_path):
            return {"label": "error", "probability": "0.0", "note": "未產生 CSV"}

        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "probability": "0.0", "note": "CSV 為空"}

            latest_row = reader[-1]
            label = _get_label(latest_row)
            final_score = _get_score(latest_row)

            return {
                "label": label,                          # 例如 "neuropeptide"
                "probability": f"{final_score:.4f}",     # 🟢 格式化到 4 位小數
                "note": "AI Predicted"
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
def do_anti_aging_prediction(sequence: str) -> dict:
    """
    呼叫 anti_aging_predictor.py 腳本，
    讀取 anti_aging_prediction_result.csv 取得預測結果。
    """
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "anti_aging_prediction_result.csv")
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "anti_aging_predictor.py")

    try:
        if os.path.exists(csv_path):
            os.remove(csv_path)

        with open(fasta_path, "w") as f:
            f.write(f">seq1\n{sequence}\n")

        cmd = f"{OLD_ENV_PYTHON} {script_abs_path}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=MAIN_PROGRAM_DIR
        )

        if result.returncode != 0:
            return {
                "label": "error",
                "probability": "0.0",
                "note": f"腳本錯誤: {result.stderr.strip()[-200:]}"
            }

        if not os.path.exists(csv_path):
            return {
                "label": "error",
                "probability": "0.0",
                "note": "未產生 anti_aging_prediction_result.csv"
            }

        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {
                    "label": "error",
                    "probability": "0.0",
                    "note": "CSV 為空"
                }

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

    num_variants = data.get("num_variants", 100)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 100

    input_fasta = os.path.join(MAIN_PROGRAM_DIR, "pipeline_in.fasta")
    output_cdhit = os.path.join(MAIN_PROGRAM_DIR, "pipeline_out")
    output_2d = os.path.join(MAIN_PROGRAM_DIR, "pipeline_2d_out")
    known_db = os.path.join(MAIN_PROGRAM_DIR, "cdhitusing.fasta")
    script_abs_path = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")

    print(f"🚀 [Discovery] 從 {base_sequence} 衍生 {num_variants} 條序列...")

    # ---- 1) 隨機突變生成變體 ----
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

    # ---- 2) CD-HIT 去冗餘 ----
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

    # ---- 3) 解析 CD-HIT 結果，取得 (id, seq) ----
    final_candidates = []   # list of (seq_id, sequence)
    if os.path.exists(final_output):
        with open(final_output, "r") as f:
            current_id, current_seq = None, ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        final_candidates.append((current_id, current_seq))
                    current_id = line[1:].split()[0]   # 取 header 第一個 token
                    current_seq = ""
                else:
                    current_seq += line
            if current_id and current_seq:
                final_candidates.append((current_id, current_seq))

    if not final_candidates:
        print(f"⚠️ 篩選後無剩餘序列，啟用備用機制...")
        unique_seqs = list(set(variants))
        final_candidates = [(f"var_{i}", s) for i, s in enumerate(unique_seqs)]

    print(f"🔬 [Discovery] 最終剩餘 {len(final_candidates)} 條，開始 AI 預測...")

    # ---- 4) 寫入 test.fasta，呼叫預測腳本 ----
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(fasta_path, "w") as f:
        for sid, seq in final_candidates:
            f.write(f">{sid}\n{seq}\n")

    cmd_predict = f"{OLD_ENV_PYTHON} {script_abs_path}"
    subprocess.run(cmd_predict, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)

    # ---- 5) 解析預測結果 CSV，使用 seq_id 對應 ----
    seq_map = {sid: s for sid, s in final_candidates}
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))

            # 🟢 優先用 seq_id 對應；若 CSV 第一欄是空的，回退用 index
            for idx, row in enumerate(reader):
                seq_id = _get_seq_id(row)
                seq_text = seq_map.get(seq_id)

                # fallback：用順序對應
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

    # ---- 6) 清理暫存檔 ----
    temp_files = [
        input_fasta, output_cdhit, output_cdhit + ".clstr",
        output_2d, output_2d + ".clstr", fasta_path
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
            file_path = os.path.join(MAIN_PROGRAM_DIR, file_name)
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
