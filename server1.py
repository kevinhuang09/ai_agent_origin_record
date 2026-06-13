import sys
import os
import csv
import subprocess
import requests
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import random
import torch
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd

OLD_ENV_PYTHON = "/home/g503/miniconda3/envs/neuropredictor-tool/bin/python"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PROGRAM_DIR = os.path.join(BASE_DIR, "mainProgram")
# 確保圖片儲存資料夾存在
IMAGE_DIR = os.path.join(MAIN_PROGRAM_DIR, "plots")
os.makedirs(IMAGE_DIR, exist_ok=True)
if MAIN_PROGRAM_DIR not in sys.path:
    sys.path.append(MAIN_PROGRAM_DIR)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── [1. 建立轉換字典] ──
# 將氨基酸轉成數字 ID (需與你訓練時的字典完全一致)
AA_TO_ID = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=1)}
# 假設 0 是用來 Padding 或是 Unknown 的
AA_TO_ID["<PAD>"] = 0  

# 將模型輸出的數字 ID 轉回結構字串
ID_TO_LABEL = {0: "C", 1: "H", 2: "E"}

# --- [預測核心：真正的把變體丟進 Predictor] ---
def do_prediction(sequence: str) -> dict:
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")

    try:
        # 🚨 關鍵 1：每次預測前，強制刪除舊的 CSV，避免讀到「上一條」的分數
        if os.path.exists(csv_path):
            os.remove(csv_path)

        os.makedirs(MAIN_PROGRAM_DIR, exist_ok=True)
        
        # 🚨 關鍵 2：將「目前的變體序列」寫入 fasta，讓你的 AI 模型讀取
        with open(fasta_path, "w") as f:
            f.write(f">agent_query\n{sequence}\n")
        
        # 執行原本的預測程式
        cmd = f"{OLD_ENV_PYTHON} predict_web.py"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)
        
        if result.returncode != 0:
            return {"label": "error", "score": "0.0", "note": f"報錯: {result.stderr.strip()[-100:]}"}
            
        if not os.path.exists(csv_path):
            return {"label": "error", "score": "0.0", "note": "未產生 CSV"}
            
        # 🚨 關鍵 3：讀取並只抓取「最後一行」，確保拿到最新分數
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return {"label": "error", "score": "0.0", "note": "CSV 為空"}
            
            latest_row = reader[-1] 
            return {
                "label": latest_row.get('Final_Label', 'unknown'), 
                "score": latest_row.get('Average_Score', '0.0'), 
                "note": "AI Predicted"
            }
    except Exception as e:
        return {"label": "error", "score": "0.0", "note": str(e)}

@app.post("/predict")
async def predict(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "No sequence"}
    res = do_prediction(sequence)
    if res["label"] == "error": return {"status": "error", "message": res["note"]}
    return {"status": "success", "prediction": res}

try:
    import ankh_service
    get_ankh_embedding = ankh_service.get_ankh_embedding
except Exception:
    def get_ankh_embedding(seq): raise Exception("Ankh 未載入")

@app.post("/ankh")
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
    if not base_sequence: return {"status": "error", "message": "空序列"}
        
    # 👇 1. 動態接收變體數量，如果 Agent 沒傳，預設才是 100
    num_variants = data.get("num_variants", 100)
    try:
        num_variants = int(num_variants)
    except ValueError:
        num_variants = 100

    input_fasta = os.path.join(MAIN_PROGRAM_DIR, "pipeline_in.fasta")
    output_cdhit = os.path.join(MAIN_PROGRAM_DIR, "pipeline_out")
    
    print(f"🚀 [Discovery] 從 {base_sequence} 衍生 {num_variants} 條序列...")

    # 👇 2. 下面的 [步驟 1: 衍生變體] 照舊，它現在會使用動態的 num_variants 來跑迴圈
    variants = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(num_variants):
        seq_list = list(base_sequence)
        mutate_times = 1 if len(seq_list) <= 1 else random.randint(1, 2)
        for _ in range(mutate_times):
            pos = random.randint(0, len(seq_list) - 1)
            seq_list[pos] = random.choice(amino_acids)
        variants.append("".join(seq_list))
    
    # [步驟 2: CD-HIT 去冗餘]
    with open(input_fasta, "w") as f:
        for i, v in enumerate(variants): f.write(f">var_{i}\n{v}\n")
    
    cmd = f"cd-hit -i {input_fasta} -o {output_cdhit} -c 0.9 -n 5"
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    final_candidates = []
    if os.path.exists(output_cdhit):
        with open(output_cdhit, "r") as f:
            current_seq = ""
            for line in f:
                if line.startswith(">"):
                    if current_seq: final_candidates.append(current_seq)
                    current_seq = ""
                else: current_seq += line.strip()
            if current_seq: final_candidates.append(current_seq)

    # 短肽防呆機制
    if not final_candidates:
        print(f"⚠️ CD-HIT 過濾失敗，啟用備用機制...")
        final_candidates = list(set(variants))

    # [步驟 3: 🔥 一次性批量預測 (Batch Prediction)]
    print(f"🔬 [Discovery] 去冗餘後剩餘 {len(final_candidates)} 條，將序列打包進行【一次性 AI 預測】...")
    
    fasta_path = os.path.join(MAIN_PROGRAM_DIR, "test.fasta")
    csv_path = os.path.join(MAIN_PROGRAM_DIR, "final_prediction_results.csv")
    
    # 1. 強制清理舊的 CSV
    if os.path.exists(csv_path): os.remove(csv_path)
        
    # 2. 將所有候選序列一次寫入 test.fasta
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(final_candidates):
            f.write(f">var_{i}\n{seq}\n")
            
    # 3. 只呼叫「一次」 AI 預測程式！(速度提升 30 倍)
    cmd = f"{OLD_ENV_PYTHON} predict_web.py"
    subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=MAIN_PROGRAM_DIR)
    
    # 4. 解析包含所有結果的 CSV
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            # 依序讀取每一行的結果，配對回原本的序列
            for i, row in enumerate(reader):
                if i < len(final_candidates):
                    results.append({
                        "seq": final_candidates[i], 
                        "label": row.get('Final_Label', 'unknown'), 
                        "score": row.get('Average_Score', '0.0')
                    })
    else:
        print("❌ AI 預測程式未產生輸出 CSV！")

    # 清理暫存檔案 (把 test.fasta 也一併清掉保持乾淨)
    for f_path in [input_fasta, output_cdhit, output_cdhit + ".clstr", fasta_path]:
        if os.path.exists(f_path): os.remove(f_path)

    # 排序並抓取前 5 名
    top_results = sorted(results, key=lambda x: float(x['score']), reverse=True)[:5]
    print(f"✅ [Discovery] 批量預測完成！選出前 5 名。")
    
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
        # 呼叫 Meta ESMFold 官方免費 API
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        headers = {"Content-Type": "text/plain"}
        
        # 發送序列字串給 API
        response = requests.post(url, data=sequence, headers=headers, timeout=60)
        
        if response.status_code == 200:
            pdb_data = response.text
            
            # 將 3D 結構資料儲存為 .pdb 檔案 (存在 mainProgram 資料夾內)
            file_name = f"esmfold_{sequence[:15]}.pdb"
            file_path = os.path.join(MAIN_PROGRAM_DIR, file_name)
            
            with open(file_path, "w") as f:
                f.write(pdb_data)
                
            return {
                "status": "success", 
                "message": "3D 結構預測完成", 
                "file_path": file_path
            }
        else:
            return {"status": "error", "message": f"ESMFold API 報錯，狀態碼: {response.status_code}"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/secondary_structure")
async def secondary_structure(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "空序列"}
    
    try:
        # --- [A. 序列轉換為 Tensor] ---
        # 1. 將字串轉換為 ID 列表 (遇到不認識的氨基酸預設給 0)
        input_ids = [AA_TO_ID.get(aa, 0) for aa in sequence]
        
        # 2. 轉換為 PyTorch Tensor，並加上 Batch 維度 -> shape 變成 [1, sequence_length]
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        # 如果你有用 GPU，記得把資料搬過去
        # input_tensor = input_tensor.to(device)

        # --- [B. 執行模型推論] ---
        # 🚨 極度重要：使用 torch.no_grad() 節省記憶體，並防止梯度計算
        # with torch.no_grad():
            # 將 Tensor 丟入模型
            # logits = model(input_tensor)  
            
            # 假設 logits 的 shape 是 [1, sequence_length, 3 (類別數)]
            # 沿著最後一個維度(dim=-1)找出機率最大的 index
            # predictions = torch.argmax(logits, dim=-1)
            
            # 把 Batch 維度去掉 (squeeze)，並轉回 Python 的 list
            # pred_list = predictions.squeeze(0).tolist()
            
        # 💡 (在你把上面真實模型解開註解前，先用這個 list 模擬過渡)
        import random
        pred_list = [random.choice([0, 1, 2]) for _ in sequence]

        # --- [C. 將預測結果轉回 H/E/C 字串] ---
        structure = "".join([ID_TO_LABEL.get(pred_id, "C") for pred_id in pred_list])
        
        h_count = structure.count("H")
        e_count = structure.count("E")
        c_count = structure.count("C")
        
        summary = f"Alpha 螺旋(H): {h_count}, Beta 摺疊(E): {e_count}, 無規捲曲(C): {c_count}"
        
        return {
            "status": "success", 
            "sequence": sequence,
            "structure": structure, 
            "summary": summary
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/full_analysis_pipeline")
async def full_analysis_pipeline(data: dict):
    sequence = data.get("sequence")
    # 1. 執行 Discovery 找到 Top 1
    # 2. 對 Top 1 執行理化繪圖 (已有的函數)
    # 3. 對 Top 1 執行 ESMFold (已有的函數)
    # 4. 回傳包含所有結果的 JSON
    return {
        "status": "success",
        "top_sequence": "...",
        "mw": "...",
        "pI": "...",
        "plot_path": "...",
        "pdb_path": "..."
    }
@app.post("/auto_research_pipeline")
async def auto_research_pipeline(data: dict):
    # 步驟 1: 執行 Discovery Pipeline 產生變體並選出第一名
    # 這裡調用您現有的 discovery 邏輯
    discovery_res = run_discovery(data["sequence"], num_variants=10)
    top_variant = discovery_res["top_results"][0]["seq"]
    
    # 步驟 2: 針對 top_variant 同時執行理化分析與 3D 預測
    # 這裡調用您現有的 analyze_physicochemical 與 esmfold 邏輯
    phys_res = analyze_physicochemical(top_variant)
    esm_res = predict_esmfold(top_variant)
    
    return {
        "status": "success",
        "original_seq": data["sequence"],
        "best_variant": top_variant,
        "variant_score": discovery_res["top_results"][0]["score"],
        "physicochemical": phys_res,
        "pdb_path": esm_res["file_path"]
    }
@app.post("/analyze_physicochemical")
async def analyze_physicochemical(data: dict):
    sequence = data.get("sequence", "").strip().upper()
    if not sequence: return {"status": "error", "message": "無效序列"}

    try:
        analysed_seq = ProteinAnalysis(sequence)
        
        # 1. 計算疏水性 (Kyte & Doolittle scale)
        # hydrophobicity = analysed_seq.protein_scale(window=5, edge=0.4, scale="kd")
        hydrophobicity = analysed_seq.protein_scale(kd, window=5, edge=0.4)
        # 2. 繪圖
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(hydrophobicity) + 1), hydrophobicity, marker='o', linestyle='-', color='b')
        plt.axhline(0, color='black', linestyle='--', alpha=0.3) # 0 線以上是疏水，以下是親水
        plt.title(f"Hydrophobicity Profile (Kyte-Doolittle) - {sequence[:10]}...")
        plt.xlabel("Amino Acid Position")
        plt.ylabel("Hydrophobicity Score")
        plt.grid(True, alpha=0.3)

        # 儲存圖片
        file_name = f"phys_plot_{sequence[:10]}.png"
        file_path = os.path.join(IMAGE_DIR, file_name)
        plt.savefig(file_path)
        plt.close() # 🚨 重要：關閉畫布釋放記憶體

        # 3. 計算一些額外數據
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

@app.api_route("/ollama/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def ollama_proxy(path: str, request: Request):
    url = f"http://127.0.0.1:11434/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None); headers.pop("content-length", None)
    try:
        response = requests.request(method=request.method, url=url, data=body, headers=headers, stream=True)
        return StreamingResponse(response.iter_content(chunk_size=1024), status_code=response.status_code)
    except:
        return {"error": "連線失敗"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)