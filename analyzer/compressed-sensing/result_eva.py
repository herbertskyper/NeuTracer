import numpy as np
import matplotlib.pyplot as plt
import os

# 文件夹路径
label_dir = './dataset/NAB/label'
result_dir = './result/NAB/result'
# save_dir = './result/tmp_tmp'
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils_detect.metrics import evaluate_result
# os.makedirs(save_dir, exist_ok=True)

# 获取所有label文件
results = []
for fname in os.listdir(label_dir):
    if not fname.endswith('.csv'):
        continue
    # print(f"Processing {fname}...")
    label_path = os.path.join(label_dir, fname)
    result_path = os.path.join(result_dir, fname.replace('_label.csv', '.csv'))
    if not os.path.exists(result_path):
        print(f"Result file not found for {fname}, skip.")
        continue

    # 读取数据
    label = np.loadtxt(label_path, dtype=int)
    result = np.loadtxt(result_path, dtype=int)
    min_len = min(len(label), len(result))
    label = label[:min_len]
    result = result[:min_len]
# expand 
# 0 = 0.524
# 1 = 0.524
# 2 = 0.5490
# 3\4\5 = 0.5490


    expand = 5
    if np.any(label == 1):
        idx = np.where(label == 1)[0]
        for i in idx:
            left = max(0, i - expand)
            right = min(len(label), i + expand + 1)
            label[left:right] = 1

    precision, recall, f1score = evaluate_result(result, label)
    # try:
    #     sample_id = int(fname.replace('service', '').replace('.csv', ''))
    # except Exception:
    sample_id = fname
    results.append((sample_id, fname, precision, recall, f1score))

    # # 画图
    # plt.figure(figsize=(16, 5))
    # plt.plot(label, label='Ground Truth', linewidth=2)
    # plt.plot(result, label='Detect Result', linewidth=2, linestyle='--')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title(f'{fname} vs {os.path.basename(result_path)}')
    # plt.legend()
    # save_path = os.path.join(save_dir, f'compare_{fname.replace(".csv", "")}.png')
    # plt.savefig(save_path)

# 按样本编号排序输出
# results.sort(key=lambda x: x[0])
for _, fname, precision, recall, f1score in results:
    print(f"{fname}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1score:.4f}")
recalls = [recall for _, _, _, recall, _ in results]
mean_recall = np.mean(recalls)
print(f"\n平均召回率: {mean_recall:.4f}")