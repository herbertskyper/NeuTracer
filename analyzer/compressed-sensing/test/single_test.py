import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from detect import detect
import warnings
warnings.filterwarnings('ignore')

def calculate_evaluation_metrics(y_true, y_pred):
    """
    计算评估指标
    """
    # 基本指标
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算额外指标
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }
    
    return metrics

def print_evaluation_results(metrics, title="评估结果"):
    """
    打印格式化的评估结果
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    print(f"精确率 (Precision):     {metrics['precision']:.4f}")
    print(f"召回率 (Recall):        {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):      {metrics['f1_score']:.4f}")
    print(f"准确率 (Accuracy):      {metrics['accuracy']:.4f}")
    print(f"特异性 (Specificity):   {metrics['specificity']:.4f}")
    
    print(f"\n混淆矩阵:")
    print(f"真正例 (TP): {metrics['true_positives']:>6}")
    print(f"假正例 (FP): {metrics['false_positives']:>6}")
    print(f"真负例 (TN): {metrics['true_negatives']:>6}")
    print(f"假负例 (FN): {metrics['false_negatives']:>6}")
    
    print(f"{'='*50}")

def test_single_service():
    """
    测试单个服务文件
    """
    print("开始单个样例测试...")
    
    # 文件路径配置
    data_file = "./dataset/a.csv"
    label_file = "./dataset/b.csv"
    output_file = "./test_output/service0_result.csv"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    if not os.path.exists(label_file):
        print(f"❌ 标签文件不存在: {label_file}")
        return False
    
    print(f"数据文件: {data_file}")
    print(f"标签文件: {label_file}")
    print(f"输出文件: {output_file}")
    
    try:
        # 检查数据文件内容
        print("\n1. 检查数据文件...")
        data_df = pd.read_csv(data_file)
        print(f"   数据形状: {data_df.shape}")
        print(f"   列名数量: {len(data_df.columns)}")
        
        # 检查标签文件内容
        print("\n2. 检查标签文件...")
        label_df = pd.read_csv(label_file, header=None)
        print(f"   标签形状: {label_df.shape}")
        print(f"   异常点数量: {label_df.iloc[:, 0].sum()} / {len(label_df)}")
        print(f"   异常率: {label_df.iloc[:, 0].sum() / len(label_df) * 100:.2f}%")
        
        # 运行异常检测
        print("\n3. 运行异常检测...")
        start_time = time.time()
        
        detect(
            data_in=data_file,
            data_out=output_file,
            row_start=0,     
            col_start=0,     
            header=1        
        )
        
        processing_time = time.time() - start_time
        print(f"   处理时间: {processing_time:.2f}秒")
        
        # 检查输出文件是否生成
        if not os.path.exists(output_file):
            print(f"❌ 输出文件未生成: {output_file}")
            return False
        
        print("   ✅ 异常检测完成")
        
        # 加载预测结果
        print("\n4. 加载预测结果...")
        predictions = pd.read_csv(output_file, header=None).iloc[:, 0].values.astype(int)
        true_labels = label_df.iloc[:, 0].values.astype(int)
        
        print(f"   预测结果形状: {predictions.shape}")
        print(f"   真实标签形状: {true_labels.shape}")
        
        # 确保长度匹配
        min_length = min(len(predictions), len(true_labels))
        if len(predictions) != len(true_labels):
            print(f"   ⚠️ 长度不匹配，截取至 {min_length} 个数据点")
            predictions = predictions[:min_length]
            true_labels = true_labels[:min_length]
        
        # 计算评估指标
        print("\n5. 计算评估指标...")
        metrics = calculate_evaluation_metrics(true_labels, predictions)
        
        # 打印详细结果
        print_evaluation_results(metrics, "service0 异常检测评估结果")
        
        # 打印数据统计
        print(f"\n数据统计:")
        print(f"总数据点数: {len(predictions)}")
        print(f"真实异常点数: {np.sum(true_labels)} ({np.sum(true_labels)/len(true_labels)*100:.2f}%)")
        print(f"预测异常点数: {np.sum(predictions)} ({np.sum(predictions)/len(predictions)*100:.2f}%)")
        print(f"处理时间: {processing_time:.2f}秒")
        
        # 保存详细结果
        result_summary = {
            'service_name': 'service0',
            'total_points': len(predictions),
            'true_anomalies': int(np.sum(true_labels)),
            'predicted_anomalies': int(np.sum(predictions)),
            'processing_time': processing_time,
            **metrics
        }
        
        summary_file = "./test_output/service0_detailed_result.csv"
        summary_df = pd.DataFrame([result_summary])
        summary_df.to_csv(summary_file, index=False)
        print(f"\n详细结果已保存到: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    print("=" * 60)
    print("单个样例测试脚本 - service0")
    print("=" * 60)
    
    success = test_single_service()
    
    if success:
        print(f"\n🎉 单个样例测试完成!")
        print(f"如果结果看起来合理，可以运行批量测试脚本。")
    else:
        print(f"\n❌ 单个样例测试失败!")
        print(f"请检查错误信息并修复问题后再运行批量测试。")

if __name__ == "__main__":
    main()