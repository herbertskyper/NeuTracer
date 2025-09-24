import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm


from utils_detect import data_process
from utils_detect.metrics import sliding_anomaly_predict
from algorithm.cluster import cluster
from algorithm.lesinn import online_lesinn
from algorithm.sampling.localized_sample import localized_sample
from algorithm.cvxpy import reconstruct
# from algorithm.ncvxpy import CvxpyReconstructor
from cvxpy.error import SolverError
from time import time

# 宏定义 - 控制是否绘制各种图表
PLOT_RAW_KPI = False             # 是否绘制原始KPI数据图
PLOT_RECONSTRUCT_KPI = False      # 是否绘制重建数据对比图
PLOT_DIFF = False                # 是否绘制差异图
PLOT_TIME_SERIES = False          # 是否绘制时间序列异常图
PLOT_CLUSTER_KPI = False          # 是否绘制聚类结果图
PLOT_SAMPLE_KPI = False         # 是否绘制采样点图

# some upper limit
max_seed = 10 ** 9 + 7
USE_NUMPY = True
COMPRESS = False

def _select_keypoints_by_curvature(win_rec: np.ndarray, budget: int, min_sep: int = 5):
    """
    从单窗口重建序列中选关键点索引（每个维度单独处理曲率，最后合并去重）
    win_rec: shape=(L, d)
    return: 索引数组（窗口内索引，升序）
    """
    L, d = win_rec.shape
    if L <= 2 or budget <= 2:
        return np.array([0, max(0, L - 1)], dtype=int)
    all_selected = set()
    for j in range(d):
        amp = win_rec[:, j]
        curv = np.zeros(L)
        if L >= 3:
            curv[1:-1] = np.abs(amp[2:] - 2 * amp[1:-1] + amp[:-2])
        selected = {0, L - 1}
        cand = np.argsort(-curv)
        for idx in cand:
            if idx in (0, L - 1):
                continue
            if all(abs(idx - s) >= min_sep for s in selected):
                selected.add(int(idx))
                if len(selected) >= budget:
                    break
        all_selected.update(selected)
    return np.array(sorted(all_selected), dtype=int)

def anomaly_score_example(source: np.array, reconstructed: np.array, percentage: int = 90, topn : int = 2) -> float:
    """
    Calculate anomaly score
    :param source: original data
    :param reconstructed: reconstructed data
    :return:
    """
    if not USE_NUMPY:
        n, d = source.shape
        d_dis = np.zeros((d,))
        for i in range(d):
            dis = np.abs(source[:, i] - reconstructed[:, i])
            dis = dis - np.mean(dis)
            d_dis[i] = np.percentile(dis, percentage)
        if d <= topn:
            return d / np.sum(1 / d_dis)
        topn_result = 1 / d_dis[np.argsort(d_dis)][-1 * topn:]
        return topn / np.sum(topn_result)
    else:
        """
        Calculate anomaly score (vectorized)
        """
        # 计算每一列的绝对误差
        dis = np.abs(source - reconstructed)  # shape=(n, d)
        # 去均值
        dis = dis - np.mean(dis, axis=0, keepdims=True)
        # 按列计算百分位数
        d_dis = np.percentile(dis, percentage, axis=0)  # shape=(d,)
        # 计算分数
        if d_dis.size <= topn:
            return d_dis.size / np.sum(1 / d_dis)
        topn_result = 1 / d_dis[np.argsort(d_dis)][-topn:]
        return topn / np.sum(topn_result)

class WindowReconstructProcess():
    """
    窗口重建工作进程
    """

    def __init__(
            self,
            # reconstructor,
            data: np.array,
            cycle: int,
            latest_windows: int,
            sample_rate: float,
            scale: float,
            rho: float,
            sigma: float,
            random_state: int,
            retry_limit: int
    ):
        """
        :param data: 原始数据的拷贝
        :param cycle: 周期
        :param latest_windows: 计算采样价值指标时参考的最近历史周期数
        :param sample_score_method: 计算采样价值指标方法
        :param sample_rate: 采样率
        :param scale: 采样参数: 等距采样点扩充倍数
        :param rho: 采样参数: 中心采样概率
        :param sigma: 采样参数: 采样集中程度
        :param random_state: 随机数种子
        :param retry_limit: 每个窗口重试的上限
        """
        super().__init__()
        # self.reconstructor = reconstructor
        self.data = data
        self.cycle = cycle
        self.latest_windows = latest_windows
        self.sample_rate = sample_rate
        self.scale = scale
        self.rho = rho
        self.sigma = sigma
        self.random_state = random_state
        self.retry_limit = retry_limit

    def sample(self, x: np.array, m: int, score: np.array, random_state: int):
        """
        取得采样的数据
        :param x: kpi等距时间序列, shape=(n,d), n是行数, d是维度
        :param m: 采样个数
        :param score: 采样点置信度
        :param random_state: 采样随机种子
        :return: 采样序列数组的时间戳和值
                timestamp: shape=(m,) 表示采样点的时间位置
                values: shape=(m,d) 表示各采样点的各维度数据值
        """
        n, d = x.shape
        # print(f"输入数据形状: ({n}, {d})")
        
        # 检查是否为单列数据
        is_single_column = (d == 1)
        
        data_mat = np.asmatrix(x)
        sample_matrix, timestamp = localized_sample(
            x=data_mat, m=m,
            score=score,
            scale=self.scale, rho=self.rho, sigma=self.sigma,
            random_state=random_state
        )
        
        # 采样中心对应的位置
        s = np.array(sample_matrix * data_mat)
        # 从这里开始修改
        timestamp = np.array(timestamp).astype(int)

        # 排序索引
        sort_idx = np.argsort(timestamp)
        timestamp_sorted = timestamp[sort_idx]
        values_sorted = s[sort_idx] if d > 1 else s[sort_idx].reshape(-1, 1)

        return timestamp_sorted, values_sorted
        
        # # 构建结果列表
        # res = []
        # for i in range(m):
        #     if is_single_column:
        #         # 单列数据时s[i]是标量数组，确保提取其中的值
        #         res.append((timestamp[i], s[i].flatten()[0] if hasattr(s[i], 'flatten') else s[i]))
        #     else:
        #         # 多列数据直接添加
        #         res.append((timestamp[i], s[i]))
        
        # # 按时间戳排序
        # res.sort(key=lambda each: each[0])
        
        # # 提取时间戳和值
        # timestamp = np.array([item[0] for item in res]).astype(int)
        # # res = np.array(res)
        # # timestamp = np.array(res[:, 0]).astype(int)
        
        # # 根据数据列数构建values数组
        # values = np.zeros((m, d))
        # for i in range(m):
        #     if is_single_column:
        #         # 单列数据特殊处理
        #         values[i, 0] = res[i][1]
        #     else:
        #         # 多列数据直接赋值
        #         values[i, :] = res[i][1]
        
        # return timestamp, values

    def window_sample_reconstruct(
            self,
            data: np.array,
            groups: list,
            score: np.array,
            random_state: int
    ):
        """
        :param data: 原始数据
        :param groups: 分组
        :param score: 这个窗口的每一个点的采样可信度
        :param random_state: 随机种子
        :return: 重建数据, 重建尝试次数
        """
        # 数据量, 维度
        n, d = data.shape
        retry_count = 0
        sample_rate = self.sample_rate
        # print("窗口大小: %d, 采样率: %f" % (n, sample_rate))
        while True:
            try:
                # t1 = time()
                timestamp, values = \
                    self.sample(
                        data,
                        int(np.round(sample_rate * n)),
                        score,
                        random_state
                    )
                rec = np.zeros(shape=(n, d))
                # t2 = time()
                # print(f"采样耗时: {t2 - t1:.2f}秒")
                for i in range(len(groups)):
                    x_re = reconstruct(
                        n, len(groups[i]), timestamp,
                        values[:, groups[i]]
                    )
                    for j in range(len(groups[i])):
                        rec[:, groups[i][j]] = x_re[:, j]
                # t3 = time()
                # print(f"重建耗时: {t3 - t2:.2f}秒")
                break
            except SolverError:
                if retry_count > self.retry_limit:
                    raise Exception(
                        'retry failed, please try higher sample rate or '
                        'window size'
                    )
                sample_rate += (1 - sample_rate) / 4
                retry_count += 1
                from sys import stderr
                stderr.write(
                    'WARNING: reconstruct failed, retry with higher '
                    'sample rate %f, retry times remain %d\n'
                    % (
                        sample_rate, self.retry_limit - retry_count)
                )
        return rec, retry_count


def detect(data_in, data_out, row_start=0, col_start=2, header=1,):
    config = 'detector-config.yml'
    with open(config, 'r', encoding='utf8') as file:
        config_dict = yaml.load(file, Loader=yaml.Loader)
    raw_data = pd.read_csv(data_in)

    # Chop down data size
    data = raw_data.iloc[row_start:raw_data.shape[0], col_start:raw_data.shape[1]]
    column_names = raw_data.columns[col_start:].tolist()

    n, d = data.shape

    # Normalize each dimension
    data = data.values
    for i in range(d):
        data[:, i] = data_process.normalization(data[:, i]) #question: should we normalize here?

    # 采样时参考的历史窗口数
    latest_windows = config_dict['detector_arguments']['latest_windows']
    # 采样率
    sample_rate = config_dict['detector_arguments']['sample_rate']
    rho = config_dict['detector_arguments']['rho']
    sigma = config_dict['detector_arguments']['sigma']
    scale = config_dict['detector_arguments']['scale']
    retry_limit = config_dict['detector_arguments']['retry_limit']
    random_state = config_dict['global']['random_state']

    t = config_dict['sample_score_method']['lesinn']['t']
    phi = config_dict['sample_score_method']['lesinn']['phi']

    # Get clustered group
    cluster_threshold = config_dict['detector_arguments']['cluster_threshold']
    windows_per_cycle = config_dict['data']['rec_windows_per_cycle']
    reconstruct_window = config_dict['data']['reconstruct']['window']
    reconstruct_stride = config_dict['data']['reconstruct']['stride']

    detect_window = config_dict['data']['detect']['window']
    detect_stride = config_dict['data']['detect']['stride']

    predict_window = config_dict['data']['predict']['window']
    predict_stride = config_dict['data']['predict']['stride']

    cycle = reconstruct_window * windows_per_cycle
    cycle_groups = []
    group_index = 0
    global COMPRESS
    # 周期开始的index
    cb = 0
    while cb < n:
        # 周期结束的index
        ce = min(n, cb + cycle)  # 一周期数据为data[cb, ce)
        # 初始化追加列表引用
        if group_index == 0:
            # 没有历史数据
            # 分组默认每个kpi一组
            init_group = []
            for i in range(d):
                init_group.append([i])
            cycle_groups.append(init_group)
        else:
            cycle_groups.append(cluster(data[cb:ce], cluster_threshold))
        group_index += 1
        cb += cycle

    # 采样 & 重建
    # reconstructor = CvxpyReconstructor(n = reconstruct_window, d=1)
    process = WindowReconstructProcess(
        # reconstructor = reconstructor,
        data=data,
        cycle=cycle,
        latest_windows=latest_windows,
        sample_rate=sample_rate,
        scale=scale, rho=rho, sigma=sigma,
        random_state=random_state,
        retry_limit=retry_limit
    )
    # 重建的数据
    reconstructed = np.zeros((n, d))
    reconstructing_weight = np.zeros((n,))
    needed_weight = np.zeros((n,))
    total_retries = 0
    win_l = 0
    win_r = 0
    pbar = tqdm(total=n)
    while win_r < n:
        # t1 = time()
        win_r = min(n, win_l + reconstruct_window)
        group = cycle_groups[win_l // cycle]
        needed_weight[win_l:win_r] += 1

        # 采样概率
        hb = max(0, win_l - latest_windows)
        latest = data[hb:win_l]
        window_data = data[win_l:win_r]
        sample_score = online_lesinn(window_data, latest, t=t, phi=phi, random_state=random_state)
        # t3 = time()
        # print(f"在线LESINN耗时: {t3 - t1:.2f}秒")
        normal_score = 1.0 / (sample_score + 1e-8)  # 避免除零
        # # normal_score = np.exp(-anomaly_score)
        normal_score = (normal_score - normal_score.min()) / (normal_score.max() - normal_score.min() + 1e-8)
        rec_window, retries = \
            process.window_sample_reconstruct(
                data=window_data,
                groups=group,
                # score=sample_score,
                score=normal_score,
                random_state=random_state * win_l * win_r % max_seed
            )
        total_retries += retries
        for index in range(rec_window.shape[0]):
            w = index + win_l
            weight = reconstructing_weight[w]
            reconstructed[w, :] = \
                (reconstructed[w, :] * weight +
                 rec_window[index]) / (weight + 1)
        reconstructing_weight[win_l:win_r] += 1
        win_l += reconstruct_stride
        # t2 = time()
        # print(f"最外层：窗口重建耗时: {t2 - t1:.2f}秒")
        pbar.update(reconstruct_stride)

    pbar.close()

    # 预测
    # 异常得分
    anomaly_score = np.zeros((n,))
    # 表示当时某个位置上被已重建窗口的数量
    anomaly_score_weight = np.zeros((n,))
    # 窗口左端点索引
    wb = 0
    while True:
        we = min(n, wb + detect_window)
        # 窗口右端点索引 窗口数据[wb, we)
        win_data = data[wb:we]              # shape=(L, d)
        win_rec  = reconstructed[wb:we]     # shape=(L, d)
        L = we - wb

        # 原有异常评分（窗口级标量）
        score = anomaly_score_example(win_data, win_rec)
        # 累积到全局 anomaly_score（原逻辑）
        for i in range(L):
            w = i + wb
            weight = anomaly_score_weight[w]
            anomaly_score[w] = (anomaly_score[w] * weight + score) / (weight + 1)
        anomaly_score_weight[wb:we] += 1
        if we >= n:
            break
        wb += detect_stride

    if COMPRESS:
        window_size = 60
        stride = 60
        anom_idx_all, anom_vals_all, anom_scores_all = [], [], []
        key_idx_all, key_vals_all = [], []
        for wb in range(0, n, stride):
            we = min(n, wb + window_size)
            win_data = data[wb:we]
            win_rec = reconstructed[wb:we]
            L = we - wb
            if L > 0:
                # 异常点：前10%残差
                residual = np.linalg.norm(win_data - win_rec, axis=1)
                k_anom = max(1, int(np.ceil(0.20 * L)))
                top_idx_local = np.argpartition(residual, -k_anom)[-k_anom:]
                top_idx_local = top_idx_local[np.argsort(-residual[top_idx_local])]
                top_idx_global = wb + top_idx_local
                anom_idx_all.extend(top_idx_global.tolist())
                anom_vals_all.extend(win_data[top_idx_local].tolist())
                anom_scores_all.extend(residual[top_idx_local].tolist())
            if L > 1:
                # 关键点：曲率法选10%
                key_budget = max(2, int(np.ceil(0.05 * L)))
                key_local = _select_keypoints_by_curvature(win_rec, key_budget, min_sep=5)
                key_global = wb + key_local
                key_idx_all.extend(key_global.tolist())
                key_vals_all.extend(win_rec[key_local].tolist())


        # 可选：压缩保存关键点与异常点（npz）
        np.savez_compressed(
            data_out.replace('.csv', '_compressed_points.npz'),
            key_idx=np.array(key_idx_all, dtype=int),
            key_vals=np.array(key_vals_all, dtype=float),
            anom_idx=np.array(anom_idx_all, dtype=int),
            anom_vals=np.array(anom_vals_all, dtype=float),
            anom_scores=np.array(anom_scores_all, dtype=float),
        )

    # 接下来使用EVT等方式确定阈值，并做出检测
    predict = sliding_anomaly_predict(score=anomaly_score, window_size=predict_window,
                                      stride=predict_stride)
    
    np.savetxt(data_out, predict, fmt="%d",delimiter=",")
    anomaly_indices = np.where(predict == 1)[0]
    print(f"检测到{len(anomaly_indices)}个异常点")
    # import os
    # np.savetxt(os.path.dirname(data_out) + '/rec.txt' , reconstructed, '%.6f', ',')
    # import os
    # base_name = os.path.splitext(os.path.basename(data_in))[0]  # 'a'
    # # 拼接目标路径
    # save_path = os.path.join(os.path.dirname(data_out), f"{base_name}.txt")
    # # 保存分数
    # np.savetxt(save_path, predict, delimiter=',')
    # np.savetxt(os.path.dirname(data_out) + '/score.txt', anomaly_score, 'int', ',')
    # np.savetxt(os.path.dirname(data_out) + '/sample_score.txt', all_sample_score, fmt='%.6f', delimiter=',')
    # print(f"异常分数范围：{anomaly_score.min()} - {anomaly_score.max()}")
    # print(f"异常分数平均值：{anomaly_score.mean()}")
    # print(f"异常分数标准差：{anomaly_score.std()}")
    # print(f"阈值：{anomaly_score.mean() + 3 * anomaly_score.std()}")
    

    from utils_detect.plot import plot_diff, plot_time_series_with_anomalies, plot_PRC
    from utils_detect.plot_subimages import plot_raw_kpi, plot_reconstruct_kpi, plot_cluster_kpi, plot_sample_kpi
    import os
    
     # 只有当任意一个绘图开关打开时才创建图像目录
    if any([PLOT_RAW_KPI, PLOT_RECONSTRUCT_KPI, PLOT_DIFF, PLOT_TIME_SERIES, PLOT_CLUSTER_KPI, PLOT_SAMPLE_KPI]):
        image_dir = os.path.join(os.path.dirname(data_out), "image")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print(f"创建目录: {image_dir}")

        # 生成图像保存路径
        base_filename = os.path.basename(data_out).replace('.csv', '')
        image_path_base = os.path.join(image_dir, base_filename)

        # 找出检测到的异常点索引
        anomaly_indices = np.where(predict == 1)[0]
        print(f"检测到{len(anomaly_indices)}个异常点")

        # 根据宏定义决定是否绘制各种图表
        if PLOT_RAW_KPI:
            from utils_detect.plot_subimages import plot_raw_kpi
            print("绘制原始KPI数据图...")
            plot_raw_kpi(data, column_names=column_names, shape=15, 
                        save_path=f"{image_path_base}_raw_kpi.png")

        if PLOT_RECONSTRUCT_KPI:
            from utils_detect.plot_subimages import plot_reconstruct_kpi
            print("绘制重建数据对比图...")
            plot_reconstruct_kpi(data, reconstructed, column_names=column_names, shape=15, 
                                save_path=f"{image_path_base}_reconstruct_kpi.png")

        if PLOT_DIFF:
            from utils_detect.plot import plot_diff
            print("绘制差异图...")
            plot_diff(data, reconstructed, pred=anomaly_indices, column_names=column_names,
                     save_path=f"{image_path_base}_diff_plot.png")

        if PLOT_TIME_SERIES:
            from utils_detect.plot import plot_time_series_with_anomalies
            print("绘制时间序列异常图...")
            plot_time_series_with_anomalies(
                data_in=data_in,
                data=data, 
                reconstructed=reconstructed, 
                anomaly_score=anomaly_score, 
                predict=predict,
                start=row_start,
                column_names=column_names,
                save_path=f"{image_path_base}_time_series.png"
            )

        if PLOT_CLUSTER_KPI and d > 1:
            from utils_detect.plot_subimages import plot_cluster_kpi
            print("绘制聚类结果图...")
            first_group = cycle_groups[0]
            plot_cluster_kpi(data[:cycle], first_group, column_names=column_names, shape=15, 
                            save_path=f"{image_path_base}_cluster_kpi.png")

        if PLOT_SAMPLE_KPI and len(data) >= reconstruct_window:
            from utils_detect.plot_subimages import plot_sample_kpi
            print("绘制采样点图...")
            window_data = data[:reconstruct_window]
            # 获取采样分数
            hb = max(0, 0 - latest_windows)
            latest_data = data[hb:0] if hb < 0 else np.zeros((0, window_data.shape[1]))
            sample_score = online_lesinn(window_data, latest_data)
            
            # 执行采样
            m = int(np.round(sample_rate * reconstruct_window))
            timestamp, _ = process.sample(window_data, m, sample_score, random_state)
            
            # 绘制采样点
            first_group = cycle_groups[0]
            plot_sample_kpi(window_data, timestamp, first_group, shape=15, 
                           save_path=f"{image_path_base}_sample_kpi.png")

        print("所有图表绘制完成!")
    else:
        print("所有绘图功能均已禁用")

    print("Done")

if __name__ == "__main__":
    row_start = 0
    col_start = 0
    header = 1
    # detect("./dataset/a1000.csv", "./result/result_a1000.csv", row_start, col_start, header)
    detect("./col1.csv", "./result/col1.csv", row_start, col_start, header)

    # import os
    # import cProfile
    # import pstats
    # p=cProfile.Profile()
    # p.enable()
    # # detect("./dataset/Dataset1/test/service0.csv", "./result/result_service0.csv", row_start, col_start, header)
    # detect("./dataset/a2.csv", "./result/result_a2.csv", row_start, col_start, header)
    # # cProfile.run('detect("./dataset/aa.csv", "./result/result.csv", row_start, col_start, header)','lingpro.txt')
    # p.disable()
    # p.dump_stats('lingpro_service.prof')
    # ps = pstats.Stats(p).sort_stats('tottime')
    # ps.print_stats()