import os
import pandas as pd
from dvd_anomaly_detector import DVDAnomalyDetector
from detect import detect
from dvd_detector import detect_dvd
# dataset1 service2 service7 service8  service11 
# dataset2 service3  service6 service16 service26  
# dataset3 service8 

def select_method(groups):
    # 规则1: 检查是否存在单个周期组包含大量指标
    for cycle_group in groups:
        for period, metrics in cycle_group.items():
            if period != 'non-periodic' and len(metrics) >= 15:
                return "detect_dvd"
    
    # 规则2: 检查强周期组数量（指标数≥4）
    strong_group_count = 0
    
    for cycle_group in groups:
        for period, metrics in cycle_group.items():
            if period == 'non-periodic':
                continue
                
            # 规则2: 强周期组
            if len(metrics) >= 4:
                strong_group_count += 1
                    
    # 决策逻辑
    if strong_group_count >= 3 :
        return "detect_dvd"
    
    return "detect"

def choose_method_by_cycle_feature(input_path, row_start=0, col_start=0, header=0, window=7200, windows_per_cycle=1):
    # 读取数据
    df = pd.read_csv(input_path, header=header)
    df = df.iloc[row_start:, col_start:]
    data = df.values
    # 取一个默认window和windows_per_cycle（可根据你的配置灵活调整）
    detector = DVDAnomalyDetector()
    cycle = window * windows_per_cycle
    groups = detector._get_cycle_feature(data, cycle, window)
    # 判断分组特征
    # 如果所有分组都是单变量（即每组长度为1），用detect，否则用detect_dvd
    # print(f"groups for {input_path}:")
    # for i, group in enumerate(groups):
    #     print(f"  cycle {i}")
    #     for k, v in group.items():
    #         if str(k) == 'non-periodic':
    #             continue
    #         print(f"    {k}: {v}")

    method = select_method(groups)
    # return method
    return "detect"

def single_detect(input_path, output_path, row_start=0, col_start=0, header=1):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # print(f"Processing {input_path} -> {output_path}")
    method = choose_method_by_cycle_feature(input_path, row_start, col_start, header)
    print(f"Auto selected method: {method}")
    if method == "detect":
        detect(input_path, output_path, row_start, col_start, header)
    else:
        detect_dvd(input_path, output_path, row_start, col_start, header)

def batch_detect(input_dir, output_dir, row_start=0, col_start=0, header=1):
    os.makedirs(output_dir, exist_ok=True)
    # print(f"Processing files in {input_dir} and saving results to {output_dir}")
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    dvd_files = []
    for fname in files:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        # print(f"Processing {input_path} -> {output_path}")
        method = choose_method_by_cycle_feature(input_path, row_start, col_start, header)
        # print(f"Auto selected method: {method}")
        if method == "detect_dvd":
            dvd_files.append(fname)
        try:
            if method == "detect":
                detect(input_path, output_path, row_start, col_start, header)
            else:
                detect_dvd(input_path, output_path, row_start, col_start, header)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")
    print("\n使用 detect_dvd 方法的样本：")
    for f in dvd_files:
        print(f)

if __name__ == "__main__":
    # 手动指定路径和参数
    mode = "batch"  # "batch" 或 "single"

    if mode == "batch":
        input_dir = "./dataset/Dataset2/test/"
        output_dir = "./result/NAB_args"
        row_start = 0
        col_start = 1
        header = 1
        batch_detect(input_dir, output_dir, row_start, col_start, header)
    else:
        input_path = "./dataset/Dataset3/test/service0.csv"
        output_path = "./result/tmp.csv"
        row_start = 0
        col_start = 0
        header = 0
        single_detect(input_path, output_path, row_start, col_start, header)