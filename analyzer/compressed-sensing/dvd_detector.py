import numpy as np
import pandas as pd
import yaml
from time import time
import warnings
from utils_detect.metrics import spot_eval
from utils_detect.metrics import evaluation
from utils_detect.data_process import normalization
from detect import anomaly_score_example
from dvd_anomaly_detector import DVDAnomalyDetector
warnings.filterwarnings('ignore')


def read_config(config: dict):
    """
    初始化全局参数
    """
    global random_state, rec_window, rec_stride, det_window, det_stride
    global rec_windows_per_cycle
    global anomaly_score_example_percentage, anomaly_distance_topn

    # 读取 anomaly_score_example 配置
    if 'anomaly_score_example' in config:
        anomaly_scoring_config = config['anomaly_score_example']
        anomaly_score_example_percentage = int(anomaly_scoring_config.get('percentage', 100))
        anomaly_distance_topn = int(anomaly_scoring_config.get('topn', 1))

    # 读取 global 配置
    if 'global' in config:
        global_config = config['global']
        random_state = int(global_config.get('random_state', 42))

    # 读取 data 配置
    data_config = config['data']
    rec_windows_per_cycle = int(data_config.get('rec_windows_per_cycle', 1))

    # reconstruct 配置
    if 'reconstruct' in data_config:
        data_rec_config = data_config['reconstruct']
        rec_window = int(data_rec_config.get('window', 1440))
        rec_stride = int(data_rec_config.get('stride', 12))

    # detect 配置
    if 'detect' in data_config:
        data_det_config = data_config['detect']
        det_window = int(data_det_config.get('window', 14))
        det_stride = int(data_det_config.get('stride', 2))

def smooth_data(df, window=3):
    for col in df.columns:
        if col == "timestamp":
            continue
        df[col] = df[col].rolling(window=window).mean().bfill().values
    return df


def detect_dvd(input_path, output_path, row_start=0, col_start=0, header=0, config_path='dvd_config.yml'):
    """
    主检测流程
    """
    # 读取配置
    with open(config_path, 'r', encoding='utf8') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    read_config(config_dict)

    # 读取数据
    df = pd.read_csv(input_path, header=header)
    df = df.iloc[row_start:, col_start:]
    df = smooth_data(df, window=18)
    data = df.values

    # 数据归一化
    n, d = data.shape
    if n < rec_window * rec_windows_per_cycle:
        raise Exception('data point count less than 1 cycle')
    for i in range(d):
        data[:, i] = normalization(data[:, i])

    detector = DVDAnomalyDetector(
        random_state=random_state,
    )
    rec = detector.reconstruct(
        data, rec_window, rec_windows_per_cycle, rec_stride
    )
    score = detector.predict(
        data, rec, det_window, det_stride
    )
    import os
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    np.savetxt(os.path.join(output_path, base + '_rec.txt'), rec, '%.6f', ',')
    np.savetxt(os.path.join(output_path, base + '_score.txt'), score, '%.6f', ',')

    proba = spot_eval(score[:1200], score)
    np.savetxt(os.path.join(output_path, base + '_proba.txt'), proba, '%d', ',')

    return rec, score, proba