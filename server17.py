import sys
import os
import csv
import glob
import time
import subprocess
import requests
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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

# 🟢 neuro_predictor 子專案
NEURO_ROOT = os.path.join(BASE_DIR, "neuro_predictor")
MAIN_PROGRAM_DIR = os.path.join(NEURO_ROOT, "mainProgram")
DATA_DIR = os.path.join(NEURO_ROOT, "data")
PREDICT_SCRIPT = os.path.join(MAIN_PROGRAM_DIR, "predict_web.py")
ML_SCORE_DIR = os.path.join(DATA_DIR, "mlScore")
PREDICT_CSV = os.path.join(ML_SCORE_DIR, "final_prediction_results.csv")
os.makedirs(ML_SCORE_DIR, exist_ok=True)
FASTA_PATH = os.path.join(DATA_DIR, "test.fasta")
os.makedirs(DATA_DIR, exist_ok=True)

# 🌱 抗老化
ANTI_AGING_DIR = os.path.join(BASE_DIR, "Anti-angiogenic")
ANTI_AGING_SCRIPT = os.path.join(ANTI_AGING_DIR, "anti_aging_predictor.py")
ANTI_AGING_FASTA = os.path.join(ANTI_AGING_DIR, "anti_aging_test.fasta")
ANTI_AGING_CSV = os.path.join(ANTI_AGING_DIR, "anti_aging_prediction_result.csv")

# 🩸 溶血性
HEMO_DIR = os.path.join(BASE_DIR, "Hemo_predictor")
HEMO_MAIN_DIR = os.path.join(HEMO_DIR, "mainProgram")
HEMO_SCRIPT = os.path.join(HEMO_MAIN_DIR, "main_predict.py")
HEMO_FASTA = os.path.join(HEMO_MAIN_DIR, "test.fasta")
HEMO_RESULT_CSV = os.path.join(HEMO_MAIN_DIR, "final_result.csv")

# 圖片輸出
IMAGE_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(IMAGE_DIR, exist_ok=True)

if MAIN_PROGRAM_DIR not in sys.path:
    sys.path.append(MAIN_PROGRAM_DIR)

# ---- 啟動時印出路徑檢查 ----
print("=" * 60)
print("🔍 路徑檢查")
print(f"  BASE_DIR           = {BASE_DIR}")
print(f"  NEURO_ROOT         = {NEURO_ROOT}            exists={os.path.isdir(NEURO_ROOT)}")
print(f"  PREDICT_SCRIPT     = {PREDICT_SCRIPT}        exists={os.path.isfile(PREDICT_SCRIPT)}")
print(f"  ANTI_AGING_SCRIPT  = {ANTI_AGING_SCRIPT}     exists={os.path.isfile(ANTI_AGING_SCRIPT)}")
print(f"  HEMO_SCRIPT        = {HEMO_SCRIPT}           exists={os.path.isfile(HEMO_SCRIPT)}")
print(f"  OLD_ENV_PYTHON     = {OLD_ENV_PYTHON}        exists={os.path.isfile(OLD_ENV_PYTHON)}")
print("=" * 60)

# ==========================================
# 🧠 Ankh 服務載入（變體生成 + Embedding 都用這個）
# ==========================================
ANKH_AVAILABLE = False
try:
    import ankh_service
    get_ankh_embedding = ankh_service.get_ankh_embedding
    ankh_generate_by_masking = ankh_service.generate_by_masking
    ANKH_AVAILABLE = True
    print("✅ Ankh 智能突變模組已載入")
except Exception as e:
    print(f"⚠️ Ankh 未載入（會退回隨機突變）: {e}")

    def get_ankh_embedding(seq):
        raise Exception("Ankh 未載入")

    def ankh_generate_by_masking(*args, **kwargs):
        raise Exception("Ankh 未載入")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ID_TO_LABEL = {0: "C", 1: "H", 2: "E"}


# ==========================================
# 🔧 共用工具函式
# ==========================================
def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _get_seq_id(row: dict) -> str:
    return (row.get('') or row.get('seq_id') or row.get('id')
            or row.get('ID') or '').strip()


def _get_score(row: dict) -> float:
    raw = (row.get('Average_Score') or row.get('lightgbm_final')
           or row.get('probability') or row.get('score') or 0.0)
    return _safe_float(raw, 0.0)


def _get_label(row: dict) -> str:
    return row.get('Final_Label') or row.get('label') or 'unknown'


def _safe_remove(path: str):
    """安全刪除檔案，不存在或失敗都不報錯"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


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

        _safe_remove(PREDICT_CSV)

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
            return {
                "label": _get_label(latest_row),
                "probability": f"{_get_score(latest_row):.4f}",
                "note": "AI Predicted"
            }
    except Exception as e:
        return {"label": "error", "probability": "0.0", "note": str(e)}


# ==========================================
# 🌱 抗老化預測核心
# ==========================================
def do_anti_aging_prediction(sequence: str) -> dict:
    try:
        if len(sequence) < 5:
            return {"label": "error", "probability": "0.0",
                    "note": f"序列長度需 ≥ 5（目前 {len(sequence)}）"}

        if not os.path.isfile(ANTI_AGING_SCRIPT):
            return {"label": "error", "probability": "0.0",
                    "note": f"找不到抗老化腳本: {ANTI_AGING_SCRIPT}"}

        _safe_remove(ANTI_AGING_CSV)

        with open(ANTI_AGING_FASTA, "w") as f:
            f.write(f">seq1\n{sequence}\n")

        cmd = f"{OLD_ENV_PYTHON} {ANTI_AGING_SCRIPT}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=ANTI_AGING_DIR
        )

        if result.returncode != 0:
            return {"label": "error", "probability": "0.0",
                    "note": f"腳本錯誤: {result.stderr.strip()[-300:]}"}

        if not os.path.exists(ANTI_AGING_CSV):
            stdout_tail = (result.stdout or "").strip()[-400:] or "(無 stdout)"
            return {"label": "error", "probability": "0.0",
                    "note": f"未產生 CSV | stdout: {stdout_tail}"}

        with open(ANTI_AGING_CSV, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "probability": "0.0", "note": "CSV 為空"}

            latest_row = reader[-1]
            return {
                "label": _get_label(latest_row),
                "probability": f"{_get_score(latest_row):.4f}",
                "note": "AI Predicted (Anti-Aging)"
            }
    except Exception as e:
        return {"label": "error", "probability": "0.0", "note": str(e)}


# ==========================================
# 🩸 溶血性預測核心
# ==========================================
def do_hemo_prediction(sequence: str) -> dict:
    try:
        if len(sequence) < 5:
            return {"label": "error", "probability": "0.0",
                    "note": f"序列長度需 ≥ 5（目前 {len(sequence)}）"}

        if not os.path.isfile(HEMO_SCRIPT):
            return {"label": "error", "probability": "0.0",
                    "note": f"找不到溶血性腳本: {HEMO_SCRIPT}"}

        # 清掉舊 CSV
        for old_csv in glob.glob(os.path.join(HEMO_MAIN_DIR, "*.csv")):
            _safe_remove(old_csv)

        with open(HEMO_FASTA, "w") as f:
            f.write(f">seq1\n{sequence}\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = HEMO_DIR + os.pathsep + env.get("PYTHONPATH", "")

        print(f"🩸 [Hemo] cwd={HEMO_MAIN_DIR} | seq={sequence}")

        result = subprocess.run(
            [OLD_ENV_PYTHON, "main_predict.py"],
            capture_output=True, text=True,
            cwd=HEMO_MAIN_DIR, env=env, timeout=300
        )

        print(f"🩸 [Hemo] returncode={result.returncode}")
        if result.stdout:
            print(f"🩸 [Hemo] stdout(tail):\n{result.stdout[-500:]}")
        if result.stderr:
            print(f"🩸 [Hemo] stderr(tail):\n{result.stderr[-500:]}")

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()[-300:]
            return {"label": "error", "probability": "0.0",
                    "note": f"腳本錯誤: {err}"}

        if not os.path.exists(HEMO_RESULT_CSV):
            stdout_tail = (result.stdout or "").strip()[-300:] or "(無 stdout)"
            return {"label": "error", "probability": "0.0",
                    "note": f"未產生 final_result.csv | {stdout_tail}"}

        df = pd.read_csv(HEMO_RESULT_CSV, index_col=0)
        if df.empty:
            return {"label": "error", "probability": "0.0",
                    "note": "final_result.csv 為空"}

        row = df.iloc[-1]
        avg_score = _safe_float(row.get("Average_Score", 0.0), 0.0)
        pred = int(_safe_float(row.get("Final_Prediction", 0), 0))

        # 自行判斷 label（main_predict.py 內標籤是 bug）
        label = "Hemolytic" if pred == 1 else "Non-Hemolytic"

        return {
            "label": label,
            "probability": f"{avg_score:.4f}",
            "note": "AI Predicted (Hemolytic)"
        }

    except subprocess.TimeoutExpired:
        return {"label": "error", "probability": "0.0",
                "note": "執行逾時（>300s）"}
    except Exception as e:
        import traceback
        traceback.print_exc()
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


@app.post("/hemo_predictor")
async def hemo_predictor(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "No sequence"}
    res = do_hemo_prediction(sequence)
    if res["label"] == "error":
        return {"status": "error", "message": res["note"]}
    return {"status": "success", "prediction": res}


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


# ==========================================
# 🧬 蛋白質發現管線（核心函式 + 端點）
# ==========================================
def _generate_variants_smart(sequence: str, num_variants: int) -> list:
    """
    🧠 智能變體生成：優先用 Ankh，失敗才退回 random
    回傳：list[str]
    """
    seq_len = len(sequence)

    # ✅ 優先：Ankh 智能突變
    if ANKH_AVAILABLE:
        try:
            # 突變數量：序列越長改越多，但限制在合理範圍
            n_mut = max(2, min(seq_len // 3, 6))
            print(f"🧠 [Discovery] 使用 Ankh 智能突變 "
                  f"(每條突變 {n_mut}/{seq_len} 位點)")

            variants = ankh_generate_by_masking(
                sequence=sequence,
                num_variants=num_variants,
                num_mutations=n_mut,
                temperature=1.5,
                top_k=12,
            )

            if variants and len(variants) > 0:
                print(f"✅ [Discovery] Ankh 成功產出 {len(variants)} 條變體")
                return variants
            else:
                print("⚠️ [Discovery] Ankh 回傳空列表，退回 random")
        except Exception as e:
            print(f"⚠️ [Discovery] Ankh 突變失敗，退回 random: {e}")

    # ❌ Fallback：純 random（不建議，但避免 pipeline 整個炸掉）
    print("🎲 [Discovery] 使用 random 突變（fallback）")
    variants = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(num_variants):
        seq_list = list(sequence)
        mutate_times = 1 if seq_len <= 1 else random.randint(1, 2)
        for _ in range(mutate_times):
            pos = random.randint(0, seq_len - 1)
            seq_list[pos] = random.choice(amino_acids)
        variants.append("".join(seq_list))
    return variants


async def _discovery_pipeline_core(sequence: str, num_variants: int) -> dict:
    """內部函式，可直接被其他端點重複利用"""
    if not sequence:
        return {"status": "error", "message": "空序列"}
    if len(sequence) < 5:
        return {"status": "error", "message": f"序列長度需 ≥ 5（目前 {len(sequence)}）"}

    if not os.path.isfile(PREDICT_SCRIPT):
        return {"status": "error", "message": f"找不到預測腳本: {PREDICT_SCRIPT}"}

    input_fasta = os.path.join(MAIN_PROGRAM_DIR, "pipeline_in.fasta")
    output_cdhit = os.path.join(MAIN_PROGRAM_DIR, "pipeline_out")
    output_2d = os.path.join(MAIN_PROGRAM_DIR, "pipeline_2d_out")
    known_db = os.path.join(MAIN_PROGRAM_DIR, "cdhitusing.fasta")

    print(f"🚀 [Discovery] 從 {sequence} 衍生 {num_variants} 條序列...")

    # ⏱️ 變體生成計時
    t_gen_start = time.time()
    gen_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏱️ 變體生成開始時間: {gen_start_str}")

    variants = _generate_variants_smart(sequence, num_variants)

    t_gen_end = time.time()
    gen_end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gen_elapsed = t_gen_end - t_gen_start
    print(f"⏱️ 變體生成結束時間: {gen_end_str}")
    print(f"⏱️ 變體生成總耗時: {gen_elapsed:.2f} 秒")

    if not variants:
        return {"status": "error", "message": "變體生成失敗（空列表）"}

    with open(input_fasta, "w") as f:
        for i, v in enumerate(variants):
            f.write(f">var_{i}\n{v}\n")

    # CD-HIT 去冗餘
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

    _safe_remove(PREDICT_CSV)

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

    # 清理
    for f_path in [input_fasta, output_cdhit, output_cdhit + ".clstr",
                   output_2d, output_2d + ".clstr", FASTA_PATH]:
        _safe_remove(f_path)

    top_results = sorted(results, key=lambda x: x['probability'], reverse=True)[:5]

    return {
        "status": "success",
        "total_generated": num_variants,
        "after_cdhit": len(final_candidates),
        "generation_seconds": round(gen_elapsed, 2),
        "generation_method": "ankh" if ANKH_AVAILABLE else "random",
        "top_results": top_results
    }


@app.post("/discovery_pipeline")
async def discovery_pipeline(data: dict):
    base_sequence = data.get("sequence", "").strip().upper()
    num_variants = data.get("num_variants", 100)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 100
    return await _discovery_pipeline_core(base_sequence, num_variants)


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


# 🟢 內部呼叫用
async def _esmfold_predict_core(sequence: str) -> dict:
    return await esmfold_predict({"sequence": sequence})


@app.post("/analyze_physicochemical")
async def analyze_physicochemical(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "無效序列"}
    try:
        analysed_seq = ProteinAnalysis(sequence)

        # 🟢 修正：對短序列做保護
        window = 5 if len(sequence) >= 5 else max(1, len(sequence))
        hydrophobicity = analysed_seq.protein_scale(kd, window=window, edge=0.4)

        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(hydrophobicity) + 1), hydrophobicity,
                 marker='o', linestyle='-', color='b')
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


# 🟢 內部呼叫用
async def _analyze_physicochemical_core(sequence: str) -> dict:
    return await analyze_physicochemical({"sequence": sequence})


@app.post("/secondary_structure")
async def secondary_structure(data: dict):
    """
    ⚠️ 警告：此端點目前是「假預測」（純隨機），尚未接真模型。
    建議未來改接 NetSurfP-2.0 / SPIDER3 / SPOT-1D 等。
    """
    sequence = data.get("sequence", "").strip().upper()
    if not sequence:
        return {"status": "error", "message": "空序列"}
    try:
        pred_list = [random.choice([0, 1, 2]) for _ in sequence]
        structure = "".join([ID_TO_LABEL.get(pred_id, "C") for pred_id in pred_list])
        summary = (f"Alpha 螺旋(H): {structure.count('H')}, "
                   f"Beta 摺疊(E): {structure.count('E')}, "
                   f"無規捲曲(C): {structure.count('C')}")
        return {
            "status": "success",
            "sequence": sequence,
            "structure": structure,
            "summary": summary,
            "warning": "⚠️ 此結果為示意用隨機輸出，尚未接真實二級結構預測模型"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/auto_research_pipeline")
async def auto_research_pipeline(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    num_variants = data.get("num_variants", 10)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 10

    if not sequence:
        return {"status": "error", "message": "未提供序列"}
    try:
        print(f"🔄 啟動一條龍分析管線，目標序列：{sequence}，數量：{num_variants}")

        # 🟢 直接呼叫核心函式，不要重複包裝
        discovery_res = await _discovery_pipeline_core(sequence, num_variants)
        if discovery_res.get("status") != "success" or not discovery_res.get("top_results"):
            return {"status": "error", "message": "變體生成失敗"}

        top_variant = discovery_res["top_results"][0]["seq"]
        variant_probability = discovery_res["top_results"][0].get("probability", 0.0)
        variant_label = discovery_res["top_results"][0].get("label", "unknown")

        phys_res = await _analyze_physicochemical_core(top_variant)
        esm_res = await _esmfold_predict_core(top_variant)

        return {
            "status": "success",
            "total_generated": discovery_res.get("total_generated"),
            "after_cdhit": discovery_res.get("after_cdhit"),
            "generation_seconds": discovery_res.get("generation_seconds"),
            "generation_method": discovery_res.get("generation_method"),
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
        response = requests.request(method=request.method, url=url, data=body,
                                    headers=headers, stream=True)
        return StreamingResponse(response.iter_content(chunk_size=1024),
                                 status_code=response.status_code)
    except Exception:
        return {"error": "連線失敗"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)
