import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- 原始日誌字串 ---
log_string = """
Epoch 1/30 - train loss: 8.7638 | val loss: 6.9381 | ppl: 1030.82
                                                                   
Epoch 2/30 - train loss: 6.7033 | val loss: 6.2065 | ppl: 495.96
                                                                   
Epoch 3/30 - train loss: 6.0818 | val loss: 5.7010 | ppl: 299.16
                                                                   
Epoch 4/30 - train loss: 5.7039 | val loss: 5.4553 | ppl: 233.99
                                                                   
Epoch 5/30 - train loss: 5.4383 | val loss: 5.2457 | ppl: 189.75
                                                                   
Epoch 6/30 - train loss: 5.1855 | val loss: 5.0729 | ppl: 159.64
                                                                   
Epoch 7/30 - train loss: 4.9982 | val loss: 4.9571 | ppl: 142.18
                                                                   
Epoch 8/30 - train loss: 4.8601 | val loss: 4.8750 | ppl: 130.97
                                                                   
Epoch 9/30 - train loss: 4.7542 | val loss: 4.7981 | ppl: 121.28
                                                                   
Epoch 10/30 - train loss: 4.6684 | val loss: 4.7381 | ppl: 114.21
                                                                   
Epoch 11/30 - train loss: 4.6021 | val loss: 4.6900 | ppl: 108.85
                                                                   
Epoch 12/30 - train loss: 4.5514 | val loss: 4.6581 | ppl: 105.43
                                                                   
Epoch 13/30 - train loss: 4.5083 | val loss: 4.6342 | ppl: 102.94
                                                                   
Epoch 14/30 - train loss: 4.4744 | val loss: 4.6109 | ppl: 100.58
                                                                   
Epoch 15/30 - train loss: 4.4458 | val loss: 4.6017 | ppl: 99.66
                                                                   
Epoch 16/30 - train loss: 4.4242 | val loss: 4.5864 | ppl: 98.14
                                                                   
Epoch 17/30 - train loss: 4.4043 | val loss: 4.5692 | ppl: 96.47
                                                                   
Epoch 18/30 - train loss: 4.3873 | val loss: 4.5582 | ppl: 95.41
                                                                   
Epoch 19/30 - train loss: 4.3742 | val loss: 4.5545 | ppl: 95.06
                                                                   
Epoch 20/30 - train loss: 4.3646 | val loss: 4.5447 | ppl: 94.14
                                                                   
Epoch 21/30 - train loss: 4.3573 | val loss: 4.5447 | ppl: 94.14
                                                                   
Epoch 22/30 - train loss: 4.3493 | val loss: 4.5392 | ppl: 93.62
                                                                   
Epoch 23/30 - train loss: 4.3438 | val loss: 4.5377 | ppl: 93.47
                                                                   
Epoch 24/30 - train loss: 4.3403 | val loss: 4.5368 | ppl: 93.39
                                                                   
Epoch 25/30 - train loss: 4.3362 | val loss: 4.5325 | ppl: 92.99
                                                                   
Epoch 26/30 - train loss: 4.3346 | val loss: 4.5294 | ppl: 92.70
                                                                   
Epoch 27/30 - train loss: 4.3319 | val loss: 4.5294 | ppl: 92.70
                                                                   
Epoch 28/30 - train loss: 4.3311 | val loss: 4.5279 | ppl: 92.56
                                                                   
Epoch 29/30 - train loss: 4.3315 | val loss: 4.5273 | ppl: 92.51
                                                                   
Epoch 30/30 - train loss: 4.3299 | val loss: 4.5276 | ppl: 92.53
"""

# --- 數據提取與處理 ---
epochs = []
train_losses = []
val_losses = []
ppl_values = []

# 正則表達式捕捉 Epoch 數字和三個 metrics
# Group 1: Epoch 數字
# Group 2: train loss
# Group 3: val loss
# Group 4: ppl
regex_pattern = r'Epoch (\d+)/\d+ - train loss: ([\d.]+)\s*\| val loss: ([\d.]+)\s*\| ppl: ([\d.]+)'

for line in log_string.split('\n'):
    match = re.search(regex_pattern, line.strip())
    if match:
        try:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
            ppl_values.append(float(match.group(4)))
        except ValueError:
            continue

# 將數據轉換為 DataFrame 以便結構化
df = pd.DataFrame({
    'Epoch': epochs,
    'Train Loss': train_losses,
    'Val Loss': val_losses,
    'PPL': ppl_values
})

# --- 繪圖設定與執行 (使用雙 Y 軸) ---
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.suptitle('Training Metrics Trend', fontsize=16)

# 設定 X 軸
ax1.set_xlabel('Epoch')
ax1.set_xticks(df['Epoch']) # 確保 X 軸標籤只顯示實際的 Epoch 數字
ax1.grid(True, linestyle='--', alpha=0.6)


# --- Y1 軸 (Losses) ---
color_loss = 'tab:blue'
ax1.set_ylabel('Loss Value', color=color_loss)
ax1.tick_params(axis='y', labelcolor=color_loss)

# 繪製 Train Loss 和 Val Loss (無 marker)
line1 = ax1.plot(df['Epoch'], df['Train Loss'], linestyle='-', color='blue', label='Train Loss')
line2 = ax1.plot(df['Epoch'], df['Val Loss'], linestyle='-', color='darkblue', label='Val Loss')


# --- Y2 軸 (PPL) ---
# 創建第二個 Y 軸
ax2 = ax1.twinx()  
color_ppl = 'tab:red'
ax2.set_ylabel('PPL (Perplexity)', color=color_ppl) 
ax2.tick_params(axis='y', labelcolor=color_ppl)

# 繪製 PPL (無 marker)
line3 = ax2.plot(df['Epoch'], df['PPL'], linestyle='-', color='red', label='PPL')


# 統一圖例 (Legend)
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

fig.tight_layout(rect=[0, 0, 1, 0.96]) # 調整佈局以容納主標題
plt.show()

# 輸出結果 (依照您的要求移除中文)
print(df)