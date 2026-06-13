import os
os.environ['JOBLIB_MULTIPROCESSING'] = '0'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
from userPackage.Package_HEPAD import HEPAD_Predict

# ===== 參數設定 =====
dataset = 'Hmpm'
model_use = '4'

pathDict = {'paramPath': f'./data/param/{dataset}/',
            'saveCsvPath': './data/mlData/new_data/',
            'modelPath': f'./data/finalModel/{dataset}/',
            'outputPath': f'./data/output/{dataset}/'}

inputPathList = ['hemoi.fasta']

# ===== 執行預測流程 =====
encapObj = HEPAD_Predict(model_use=model_use, pathDict=pathDict)
encapObj.loadData(inputDataList=inputPathList)
encapObj.featureEncode()
encapObj.doPredict()

# ===== 整理輸出格式 =====
# 取得機率向量（每個模型的預測機率）
probDf = encapObj.probVectorDf.copy()

# 重新命名欄位（加上 _final 後綴）
probDf.columns = [f'{col}_final' for col in probDf.columns]

# 計算平均分數
probDf['Average_Score'] = probDf.mean(axis=1)

# 根據平均分數做最終預測（>= 0.5 為陽性）
probDf['Final_Prediction'] = (probDf['Average_Score'] >= 0.5).astype(int)

# 加上文字標籤
probDf['Final_Label'] = probDf['Final_Prediction'].map({
    1: 'hemolytic',
    0: 'non-hemolytic'
})

# ===== 儲存到與 main_predict.py 同一層 =====
output_filename = 'anti_hemo_prediction_result.csv'
probDf.to_csv(output_filename, index=True)

print(f'\n預測完成！結果已儲存至：{os.path.abspath(output_filename)}')
print(f'\n=== 預測結果預覽 ===')
print(probDf)
