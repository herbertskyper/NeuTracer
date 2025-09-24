import matplotlib.pyplot as plt


def plot_PRC(precision, recall, save_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.plot(recall, precision)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
def plot_diff(source, reconstruct, true=[], pred=[], title=[],column_names=None,  save_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    
    ny, nx = reconstruct.shape
    if nx > 1:
        f, ax = plt.subplots(nx, 1, figsize=(50, 4*nx))
        plt.subplots_adjust(hspace=0.5)  # 增加垂直间距
        for i in range(nx):
            kpi_name = column_names[i] if column_names and i < len(column_names) else title[i] if i < len(title) else f'Dimension {i+1}'

            ax[i].plot(source[:,i], label='Raw KPI')
            ax[i].plot(reconstruct[:,i], label='Reconstruct KPI', color='red')
            ax[i].legend()

            if len(title) > 0:
                ax[i].set_title(str(title[i]))
            else: 
                ax[i].set_title(kpi_name)
                
            if len(true) > 0:
                # ax[i].scatter(true, source[true,i], s=5, label='true', alpha=0.5, color='red')
                ax[i].vlines(true, 0.5, 1, colors='red', linestyles='-')
                
            # plt.legend()
            if len(pred) > 0:
                # ax[i].scatter(pred, source[pred,i], s=5, label='pred', alpha=0.5, color='green')
                ax[i].vlines(pred, 0, 0.5, colors='#3de1ad', linestyles='-')
    else:
        plt.figure(figsize=(50, 2))
        plt.plot(source[:, 0], label='source')
        plt.plot(reconstruct[:,0], label='Reconstruct KPI', color='red')
        plt.legend()
        if len(title) > 0:
            plt.title(str(title[0]))
        if len(true) > 0:
            # ax[i].scatter(true, source[true,i], s=5, label='true', alpha=0.5, color='red')
            plt.vlines(true, 0.5, 1, colors='red', linestyles='-')
        if len(pred) > 0:
            # ax[i].scatter(pred, source[pred,i], s=5, label='pred', alpha=0.5, color='green')
            plt.vlines(pred, 0, 0.5, colors='#3de1ad', linestyles='-')
    # plt.savefig( FILE_NAME + 'error.png' )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_time_series_with_anomalies(data_in, data, reconstructed, anomaly_score, predict, start=0, column_names=None, save_path=None, shape=30):
    """
    绘制带有时间轴的多图表视图，包括原始数据、重建数据和异常分数
    使用与其他KPI绘图函数相同的风格
    
    参数:
    data_in (str): 原始数据文件路径，用于提取时间戳
    data (np.ndarray): 原始数据数组，形状为(n, d)
    reconstructed (np.ndarray): 重建数据数组，形状为(n, d)
    anomaly_score (np.ndarray): 异常分数，形状为(n,)
    predict (np.ndarray): 异常检测结果，0表示正常，1表示异常
    start (int): 数据起始索引
    save_path (str, optional): 图表保存路径，如果为None则不保存
    shape (int): 图表尺寸，默认为30
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import matplotlib
    
    # 设置字体和样式
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    
    # 读取原始数据文件以获取时间戳
    orig_data = pd.read_csv(data_in)
    n, d = data.shape
    
    # 确保索引范围有效
    if start >= len(orig_data):
        start = 0
    end = min(start + n, len(orig_data))
    
    # 获取时间戳
    try:
        timestamps = orig_data.iloc[start:end, 0].values
        dates = [datetime.strptime(str(ts), '%Y-%m-%d %H:%M:%S') for ts in timestamps]
        use_dates = True
    except (ValueError, TypeError) as e:
        print(f"Unable to parse date format, using indices instead: {e}")
        dates = range(n)
        use_dates = False
    
    # 创建三个子图
    fig, ax = plt.subplots(3, 1, figsize=(shape, shape), sharex=True)
    
    # 使用更多颜色区分不同维度
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # 1. 绘制原始数据
    y_min, y_max = float('inf'), float('-inf')
    for i in range(d):
        label = column_names[i] if column_names and i < len(column_names) else f'Dimension {i+1}'
        ax[0].plot(dates, data[:, i], label=label, color=colors[i % len(colors)])
        y_min = min(y_min, np.min(data[:, i]))
        y_max = max(y_max, np.max(data[:, i]))
    
    # 设置纵轴范围，添加一点边距
    padding = (y_max - y_min) * 0.1
    ax[0].set_ylim([y_min - padding, y_max + padding])
    ax[0].legend(loc='upper right')
    
    # # 添加标题和图例
    # if d <= 3:  # 当维度较少时放在图例中
    #     ax[0].legend(loc='upper right')
    ax[0].set_title('Original Data', fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    
    # 2. 绘制重建数据
    y_min, y_max = float('inf'), float('-inf')
    for i in range(d):
        label = column_names[i] if column_names and i < len(column_names) else f'Dimension {i+1}'
        ax[1].plot(dates, reconstructed[:, i], label=label, 
                  color=colors[i % len(colors)], linestyle='--')
        y_min = min(y_min, np.min(reconstructed[:, i]))
        y_max = max(y_max, np.max(reconstructed[:, i]))
    
    # 设置纵轴范围，添加一点边距
    padding = (y_max - y_min) * 0.1
    ax[1].set_ylim([y_min - padding, y_max + padding])
    
    # 添加标题和图例
    # if d <= 3:
    #     ax[1].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[1].set_title('Reconstructed data', fontweight='bold')
    ax[1].grid(True, alpha=0.3)
    
    # 3. 绘制异常分数
    ax[2].plot(dates, anomaly_score, label='Anomaly Score', color='blue')
    threshold = anomaly_score.mean() + 3 * anomaly_score.std()
    ax[2].axhline(y=threshold, color='red', linestyle='--', 
                 label=f'Threshold ({threshold:.6f})')
    
    # 标记异常点
    anomaly_indices = np.where(predict == 1)[0]
    if len(anomaly_indices) > 0:
        # 确保异常索引在有效范围内
        valid_indices = [i for i in anomaly_indices if i < len(dates)]
        if valid_indices:
            anomaly_dates = [dates[i] for i in valid_indices]
            anomaly_scores = [anomaly_score[i] for i in valid_indices]
            ax[2].scatter(anomaly_dates, anomaly_scores, color='red', marker='o', s=50, 
                         label=f'Detected Anomalies ({len(valid_indices)})')
    
    ax[2].set_title(f'Anomaly Scores and Detection Results', fontweight='bold')
    ax[2].legend(loc='upper right')
    ax[2].grid(True, alpha=0.3)
    
    # 设置横轴格式
    if use_dates:
        # 根据数据时间跨度选择合适的日期格式
        if (dates[-1] - dates[0]).days > 365:  # 超过一年
            date_format = '%Y-%m'
        elif (dates[-1] - dates[0]).days > 30:  # 超过一个月
            date_format = '%m-%d'
        else:  # 短期数据
            date_format = '%m-%d %H:%M'
            
        # 设置适当数量的刻度，避免过于拥挤
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.DateFormatter(date_format)
        ax[2].xaxis.set_major_locator(locator)
        ax[2].xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()  # 自动调整日期标签角度
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # 调整子图之间的间距
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # print(f"时间序列异常图表已保存到: {save_path}")
        plt.close()
    else:
        plt.show()