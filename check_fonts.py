import tkinter as tk
from tkinter import font as tkfont

root = tk.Tk()
root.withdraw()

fonts = sorted(tkfont.families())
print(f"📊 Tkinter 看得到 {len(fonts)} 個字體\n")

print("🔍 含 kai / 楷 / MOE / CJK / Noto 的字體：")
for f in fonts:
    low = f.lower()
    if any(k in low for k in ["kai", "moe", "cjk", "noto", "ming", "hei"]) or "楷" in f:
        print(f"  ✓ {repr(f)}")  # 用 repr 確保看到精確名稱

root.destroy()
