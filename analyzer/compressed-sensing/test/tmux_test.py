import os
import pandas as pd
import numpy as np
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from detect import detect
from utils_detect.metrics import evaluate_result
import warnings
warnings.filterwarnings('ignore')

def test_single_file(data_file, label_file, output_file):
    service_name = os.path.splitext(os.path.basename(data_file))[0]
    try:
        if not (os.path.exists(data_file) and os.path.exists(label_file)):
            return {'service_name': service_name, 'status': 'FileNotFound'}

        data_df = pd.read_csv(data_file)
        label_df = pd.read_csv(label_file, header=None)
        detect(data_in=data_file, data_out=output_file, row_start=0, col_start=0, header=1)
        if not os.path.exists(output_file):
            return {'service_name': service_name, 'status': 'DetectFailed'}

        pred = pd.read_csv(output_file, header=None).iloc[:, 0].values.astype(int)
        label = label_df.iloc[:, 0].values.astype(int)
        min_len = min(len(pred), len(label))
        pred, label = pred[:min_len], label[:min_len]

        p, r, f = evaluate_result(pred, label)
        return {
            'service_name': service_name,
            'total_points': min_len,
            'true_anomalies': int(np.sum(label)),
            'precision': p, 'recall': r, 'f1_score': f,
            'status': 'Success'
        }
    except Exception as e:
        return {'service_name': service_name, 'status': f'Error: {str(e)}'}

def test_dataset_threaded(dataset_name, test_dir, label_dir, output_dir, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    filtered_files = []
    skip_ids = {1, 4, 6, 10, 15, 16,17,18,19,20,21,22,23,24,25,26, 27} if dataset_name.lower() == "dataset1" else set()
    max_files = 16 if dataset_name.lower() in {"dataset2", "dataset3"} else None

    for data_file in csv_files:
        base = os.path.splitext(os.path.basename(data_file))[0]
        if base.startswith("service"):
            try:
                idx = int(base.replace("service", ""))
            except ValueError:
                continue
            if idx in skip_ids: continue
            filtered_files.append(data_file)
            if max_files and len(filtered_files) >= max_files: break

    summary_file = os.path.join(output_dir, f"{dataset_name}_summary.csv")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(summary_file, "w") as f_summary:
        futures = []
        for data_file in filtered_files:
            service_name = os.path.splitext(os.path.basename(data_file))[0]
            label_file = os.path.join(label_dir, f"{service_name}.csv")
            output_file = os.path.join(output_dir, f"{service_name}_result.csv")
            futures.append(executor.submit(test_single_file, data_file, label_file, output_file))
        header_written = False
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            df = pd.DataFrame([result])
            if not header_written:
                df.to_csv(f_summary, index=False, header=True)
                header_written = True
            else:
                df.to_csv(f_summary, index=False, header=False)
            print(f"{result['service_name']}: {result['status']} | F1: {result.get('f1_score', 0):.4f}")

    print(f"结果已保存到: {summary_file}")
    return results

def main():
    datasets = [
        {'name': 'Dataset1', 'test_dir': './dataset/Dataset1/test', 'label_dir': './dataset/Dataset1/test_label', 'output_dir': './test_output/Dataset1'},
        {'name': 'Dataset2', 'test_dir': './dataset/Dataset2/test', 'label_dir': './dataset/Dataset2/test_label', 'output_dir': './test_output/Dataset2'},
        {'name': 'Dataset3', 'test_dir': './dataset/Dataset3/test', 'label_dir': './dataset/Dataset3/test_label', 'output_dir': './test_output/Dataset3'}
    ]
    for ds in datasets:
        print(f"\n=== 测试 {ds['name']} ===")
        test_dataset_threaded(ds['name'], ds['test_dir'], ds['label_dir'], ds['output_dir'], max_workers=8)

if __name__ == "__main__":
    main()