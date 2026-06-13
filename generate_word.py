# 強制指定字型檔案路徑
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = 'C:/Windows/Fonts/msjh.ttc'  # 微軟正黑體
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>服務層級</b>', '<b>埠口 (Port)</b>', '<b>功能說明</b>'],
        fill_color='#2E4057',
        font=dict(color='white', size=14),
        align='center', height=45
    ),
    cells=dict(
        values=[
            ['Ollama 推論引擎', 'FastAPI 閘道', 'Cloudflare Tunnel'],
            ['<b>8800</b>', '<b>11434</b>', '加密通道'],
            ['本地 LLM 模型推論服務（Gemma / Llama3 等），負責 token 生成與模型管理',
             '統一 API Gateway，整合神經胜肽預測、Ankh、ESMFold、理化分析等工具端點',
             '無固定 IP 的安全反向代理，將內部 11434 / 8800 服務加密對外']
        ],
        fill_color=[['#F4F6F8', '#E8ECEF', '#F4F6F8']],
        font=dict(color=['black', '#FF6B35', 'black'], size=13),
        align=['center', 'center', 'left'],
        height=55
    ),
    columnwidth=[200, 130, 450]
)])

fig.update_layout(
    title=dict(text='<b>服務埠口配置總覽</b>', x=0.5,
               font=dict(size=18, color='#2E4057')),
    width=900, height=320,
    margin=dict(l=20, r=20, t=60, b=20)
)

# 需安裝 kaleido：pip install kaleido
fig.write_image('port_config.png', scale=2)
fig.show()
