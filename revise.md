*** 可以增加預測模型是用lightgbm來做預測
*** print出原始peptide ， 可以在多一些特性 1. 帶電
*** 針對2D, 3D設計50~80 length peptide

### 收集所有問題整理為一份報告(pdf, word)
### 儲存3D圖片 放到報告中

我這裡有100條序列 幫我預測前10有可能是neuropeptide 並做優化
優化 ：
(改變胺基酸) compare 如果 高輸出(origin vs ankh peptide)

幫我生成結構與peptide最像的序列
用ankh生100條
predict 100條 這邊可以不用cdhit!!!!!
輸出結構最相似的 10條
結構比較工具tm align(學這個tool) 

幫我優化這條neuropeptide 優化完的結構要跟原始結構類似
(rmsd < 3)

### 可以指定masking % 