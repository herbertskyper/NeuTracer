#pragma once
#include <cstdint>
#include <map>
#include <vector>

#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"
#include <cmath>
#include <cuda_runtime.h>
#include <deque>


namespace NeuTracer {

struct FragmentationMetrics {
  // 核心指标1: 外部碎片化
  double external_fragmentation_ratio = 0.0; // 外部碎片化率
  uint64_t free_pages = 0;                   // 总可用页数
  uint64_t free_blocks_suitable = 0;         // 适合分配的空闲块数
  double kernel_unusable_index = 0.0;        // 内核不可用指数

  // 核心指标2: 分配模式评估
  double small_allocation_ratio = 0.0;   // 小分配比例
  double allocation_size_variance = 0.0; // 分配大小方差 (标准化)

  // 核心指标3: 空间效率
  double utilization_efficiency = 0.0;   // 空间利用效率
  uint64_t largest_contiguous_space = 0; // 最大连续空间
  double large_gap_ratio = 0.0;          // 大间隙比例

  // 基础统计
  uint64_t total_gap_size = 0;
  uint64_t total_address_space = 0;
  size_t gap_count = 0;
  double avg_allocation_size = 0.0;

  std::vector<uint64_t> gaps;
  std::vector<uint64_t> allocation_sizes;
};

struct FragmentationPrediction {
  std::vector<double> predicted_scores; // 预测的未来几个时间步分数
  std::vector<std::string>
      time_labels;           // 时间标签（如"下一次", "2步后", "3步后"）
  double confidence = 0.0;   // 预测置信度 (0-1)
  double trend_factor = 0.0; // 整体趋势因子
  double volatility = 0.0;   // 波动性指标
  std::string risk_level = "UNKNOWN"; // 风险等级
  std::vector<std::string> warnings;  // 预警信息
};
struct HistoricalMetrics {
  std::deque<double> scores; // 只保留历史分数用于趋势计算
  static const size_t MAX_HISTORY_SIZE = 20;

  void addRecord(double score) {
    scores.push_back(score);
    if (scores.size() > MAX_HISTORY_SIZE) {
      scores.pop_front();
    }
  }

  bool hasEnoughData() const { return scores.size() >= 3; }
};
// struct HistoricalMetrics {
//   std::deque<double> scores;               // 历史分数队列
//   std::deque<double> external_frag_ratios; // 外部碎片化率历史
//   std::deque<double> kernel_indices;       // 内核不可用指数历史
//   std::deque<double> small_alloc_ratios;   // 小分配比例历史
//   std::deque<double> large_gap_ratios;     // 大间隙比例历史
//   std::deque<size_t> allocation_counts;    // 分配数量历史
//   std::chrono::steady_clock::time_point last_update;

//   static const size_t MAX_HISTORY_SIZE = 20; // 保留最近20次记录

//   void addRecord(const FragmentationMetrics &metrics, double score) {
//     auto now = std::chrono::steady_clock::now();

//     // 添加新数据
//     scores.push_back(score);
//     external_frag_ratios.push_back(metrics.external_fragmentation_ratio);
//     kernel_indices.push_back(metrics.kernel_unusable_index);
//     small_alloc_ratios.push_back(metrics.small_allocation_ratio);
//     large_gap_ratios.push_back(metrics.large_gap_ratio);
//     allocation_counts.push_back(metrics.allocation_sizes.size());

//     // 保持队列大小
//     if (scores.size() > MAX_HISTORY_SIZE) {
//       scores.pop_front();
//       external_frag_ratios.pop_front();
//       kernel_indices.pop_front();
//       small_alloc_ratios.pop_front();
//       large_gap_ratios.pop_front();
//       allocation_counts.pop_front();
//     }

//     last_update = now;
//   }

//   bool hasEnoughData() const {
//     return scores.size() >= 3; // 至少需要3个数据点进行预测
//   }
// };

class TimeSeriesPredictor {
private:
  struct TimeSeriesSample {
    std::vector<double> features; // 当前时刻的特征
    double score;                 // 当前时刻的分数
    uint64_t timestamp;           // 时间戳
  };

  std::deque<TimeSeriesSample> time_series_data_;
  static const size_t MAX_HISTORY = 50;     // 最大历史记录
  static const size_t WINDOW_SIZE = 5;      // 时间窗口大小
  static const size_t PREDICTION_STEPS = 3; // 预测未来3个时间步

  // 线性回归权重
  std::vector<std::vector<double>> feature_weights_; // 每个预测步的特征权重
  std::vector<double> bias_terms_;                   // 偏置项
  bool is_trained_ = false;

  std::vector<double> feature_means_;
  std::vector<double> feature_stds_;
  double score_mean_ = 0.0;
  double score_std_ = 1.0;
  bool normalization_computed_ = false;

public:
  void addSample(const FragmentationMetrics &metrics, double score) {
    // 提取特征向量
    std::vector<double> features = {metrics.external_fragmentation_ratio,
                                    metrics.kernel_unusable_index,
                                    metrics.small_allocation_ratio,
                                    metrics.allocation_size_variance,
                                    metrics.large_gap_ratio,
                                    metrics.utilization_efficiency};

    TimeSeriesSample sample{
        features, score,
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count())};

    time_series_data_.push_back(sample);

    // 限制历史数据大小
    if (time_series_data_.size() > MAX_HISTORY) {
      time_series_data_.pop_front();
    }

    // 重新训练模型
    if (time_series_data_.size() >= WINDOW_SIZE + PREDICTION_STEPS) {
      trainModel();
    }
  }
  std::vector<double> predictFutureScores(const FragmentationMetrics &current_metrics) {
    std::vector<double> predictions(PREDICTION_STEPS, -1.0);

    if (!is_trained_ || time_series_data_.size() < WINDOW_SIZE || !normalization_computed_) {
        return predictions;
    }

    // 构建当前特征向量（归一化）
    std::vector<double> current_features = {
        current_metrics.external_fragmentation_ratio,
        current_metrics.kernel_unusable_index,
        current_metrics.small_allocation_ratio,
        current_metrics.allocation_size_variance,
        current_metrics.large_gap_ratio,
        current_metrics.utilization_efficiency};
    
    auto normalized_current_features = normalizeFeatures(current_features);

    // 获取最近的时间窗口数据（归一化）
    std::vector<std::vector<double>> window_features;
    std::vector<double> window_scores;

    for (size_t i = time_series_data_.size() - WINDOW_SIZE; i < time_series_data_.size(); i++) {
        window_features.push_back(normalizeFeatures(time_series_data_[i].features));
        window_scores.push_back(normalizeScore(time_series_data_[i].score));
    }

    window_features.push_back(normalized_current_features);

    // 预测未来每个时间步
    for (size_t step = 0; step < PREDICTION_STEPS; step++) {
        if (step >= feature_weights_.size() || step >= bias_terms_.size()) {
            predictions[step] = denormalizeScore(0.0); // 使用均值作为默认预测
            continue;
        }
        
        double prediction = bias_terms_[step];

        // 使用时间窗口内的所有特征
        for (size_t t = 0; t < window_features.size(); t++) {
            for (size_t f = 0; f < window_features[t].size(); f++) {
                size_t weight_idx = t * window_features[t].size() + f;
                if (weight_idx < feature_weights_[step].size()) {
                    prediction += feature_weights_[step][weight_idx] * window_features[t][f];
                }
            }
        }

        // 使用历史分数信息
        for (size_t t = 0; t < window_scores.size(); t++) {
            size_t score_weight_idx = window_features.size() * 6 + t;
            if (score_weight_idx < feature_weights_[step].size()) {
                prediction += feature_weights_[step][score_weight_idx] * window_scores[t];
            }
        }

        // 反归一化并限制范围
        double denormalized_prediction = denormalizeScore(prediction);
        predictions[step] = std::max(0.0, std::min(100.0, denormalized_prediction));

        // 将预测结果加入窗口，用于预测下一个时间步
        if (step < PREDICTION_STEPS - 1) {
            window_scores.push_back(prediction); // 使用归一化的预测值
            if (window_scores.size() > WINDOW_SIZE) {
                window_scores.erase(window_scores.begin());
            }
            if (window_features.size() > WINDOW_SIZE + 1) {
                window_features.erase(window_features.begin());
            }
            window_features.push_back(normalized_current_features);
        }
    }

    return predictions;
}

  double getConfidence() const {
    if (!is_trained_ ||
        time_series_data_.size() < WINDOW_SIZE + PREDICTION_STEPS) {
      return 0.3;
    }

    // 基于历史预测误差计算置信度
    double total_error = 0.0;
    int error_count = 0;

    // 在历史数据上验证模型
    for (size_t i = WINDOW_SIZE;
         i < time_series_data_.size() - PREDICTION_STEPS; i++) {
      // 构建训练窗口
      std::vector<std::vector<double>> window_features;
      std::vector<double> window_scores;

      for (size_t j = i - WINDOW_SIZE; j < i; j++) {
        window_features.push_back(time_series_data_[j].features);
        window_scores.push_back(time_series_data_[j].score);
      }

      // 预测并计算误差
      for (size_t step = 0;
           step < PREDICTION_STEPS && i + step < time_series_data_.size();
           step++) {
        double prediction = bias_terms_[step];

        for (size_t t = 0; t < window_features.size(); t++) {
          for (size_t f = 0; f < window_features[t].size(); f++) {
            size_t weight_idx = t * window_features[t].size() + f;
            if (weight_idx < feature_weights_[step].size()) {
              prediction +=
                  feature_weights_[step][weight_idx] * window_features[t][f];
            }
          }
        }

        for (size_t t = 0; t < window_scores.size(); t++) {
          size_t score_weight_idx = window_features.size() * 6 + t;
          if (score_weight_idx < feature_weights_[step].size()) {
            prediction +=
                feature_weights_[step][score_weight_idx] * window_scores[t];
          }
        }

        double actual = time_series_data_[i + step].score;
        total_error += std::abs(prediction - actual);
        error_count++;
      }
    }

    if (error_count == 0)
      return 0.5;

    double mae = total_error / error_count; // 平均绝对误差
    return std::max(0.1, 1.0 - mae / 50.0); // 转换为置信度
  }

  bool hasEnoughData() const {
    return time_series_data_.size() >= WINDOW_SIZE + PREDICTION_STEPS;
  }

private:
  void computeNormalizationParams() {
    if (time_series_data_.empty()) return;
    
    size_t n_features = time_series_data_[0].features.size();
    feature_means_.assign(n_features, 0.0);
    feature_stds_.assign(n_features, 0.0);
    
    // 计算特征均值
    for (const auto& sample : time_series_data_) {
        for (size_t i = 0; i < n_features; i++) {
            feature_means_[i] += sample.features[i];
        }
        score_mean_ += sample.score;
    }
    
    for (size_t i = 0; i < n_features; i++) {
        feature_means_[i] /= time_series_data_.size();
    }
    score_mean_ /= time_series_data_.size();
    
    // 计算标准差
    for (const auto& sample : time_series_data_) {
        for (size_t i = 0; i < n_features; i++) {
            feature_stds_[i] += std::pow(sample.features[i] - feature_means_[i], 2);
        }
        score_std_ += std::pow(sample.score - score_mean_, 2);
    }
    
    for (size_t i = 0; i < n_features; i++) {
        feature_stds_[i] = std::sqrt(feature_stds_[i] / time_series_data_.size());
        if (feature_stds_[i] < 1e-8) feature_stds_[i] = 1.0; // 避免除零
    }
    score_std_ = std::sqrt(score_std_ / time_series_data_.size());
    if (score_std_ < 1e-8) score_std_ = 1.0;
    
    normalization_computed_ = true;
}

std::vector<double> normalizeFeatures(const std::vector<double>& features) const {
    if (!normalization_computed_ || features.size() != feature_means_.size()) {
        return features;
    }
    
    std::vector<double> normalized(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        normalized[i] = (features[i] - feature_means_[i]) / feature_stds_[i];
    }
    return normalized;
}

double normalizeScore(double score) const {
    if (!normalization_computed_) return score;
    return (score - score_mean_) / score_std_;
}

double denormalizeScore(double normalized_score) const {
    if (!normalization_computed_) return normalized_score;
    return normalized_score * score_std_ + score_mean_;
}

void trainModel() {
    if (time_series_data_.size() < WINDOW_SIZE + PREDICTION_STEPS)
        return;

    // 计算归一化参数
    computeNormalizationParams();
    
    // 构建训练数据（使用归一化）
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;
    Y.resize(PREDICTION_STEPS);

    for (size_t i = std::max(static_cast<size_t>(0),(time_series_data_.size() - PREDICTION_STEPS - WINDOW_SIZE)); i < time_series_data_.size() - PREDICTION_STEPS + 1; i++) {
        std::vector<double> input_features;
        
        // 添加归一化的特征
        for (size_t j = i - WINDOW_SIZE; j < i; j++) {
            auto normalized_features = normalizeFeatures(time_series_data_[j].features);
            for (double feature : normalized_features) {
                input_features.push_back(feature);
            }
        }
        
        // 添加归一化的历史分数
        for (size_t j = i - WINDOW_SIZE; j < i; j++) {
            input_features.push_back(normalizeScore(time_series_data_[j].score));
        }
        
        X.push_back(input_features);
        
        // 输出：归一化的未来分数
        for (size_t step = 0; step < PREDICTION_STEPS; step++) {
            if (i + step < time_series_data_.size()) {
                Y[step].push_back(normalizeScore(time_series_data_[i + step].score));
            }
        }
    }

    if (X.empty()) return;

    feature_weights_.resize(PREDICTION_STEPS);
    bias_terms_.resize(PREDICTION_STEPS);

    for (size_t step = 0; step < PREDICTION_STEPS; step++) {
        if (Y[step].size() != X.size()) continue;

        size_t n_features = X[0].size();
        std::vector<double> weights(n_features, 0.0);
        double bias = 0.0;

        // 使用更小的学习率和更少的迭代次数
        const double learning_rate = 0.01;  // 减小学习率
        const int iterations = 500;          // 减少迭代次数
        
        double prev_loss = std::numeric_limits<double>::max();
        int no_improvement_count = 0;
        
        for (int iter = 0; iter < iterations; iter++) {
            std::vector<double> grad_weights(n_features, 0.0);
            double grad_bias = 0.0;
            double total_loss = 0.0;

            for (size_t i = 0; i < X.size(); i++) {
                // 预测
                double prediction = bias;
                for (size_t j = 0; j < n_features; j++) {
                    prediction += weights[j] * X[i][j];
                }

                // 误差
                double error = prediction - Y[step][i];
                total_loss += error * error;

                // 梯度计算
                grad_bias += error;
                for (size_t j = 0; j < n_features; j++) {
                    grad_weights[j] += error * X[i][j];
                }
            }
            
            total_loss /= X.size();
            
            // 早停机制
            if (total_loss >= prev_loss) {
                no_improvement_count++;
                if (no_improvement_count > 20) break;
            } else {
                no_improvement_count = 0;
            }
            prev_loss = total_loss;

            // 更新权重（添加梯度裁剪）
            double grad_norm = std::sqrt(grad_bias * grad_bias);
            for (size_t j = 0; j < n_features; j++) {
                grad_norm += grad_weights[j] * grad_weights[j];
            }
            grad_norm = std::sqrt(grad_norm);
            
            // 梯度裁剪
            const double max_grad_norm = 1.0;
            if (grad_norm > max_grad_norm) {
                double scale = max_grad_norm / grad_norm;
                grad_bias *= scale;
                for (size_t j = 0; j < n_features; j++) {
                    grad_weights[j] *= scale;
                }
            }

            bias -= learning_rate * grad_bias / X.size();
            for (size_t j = 0; j < n_features; j++) {
                weights[j] -= learning_rate * grad_weights[j] / X.size();
                
                // 权重裁剪，防止爆炸
                weights[j] = std::max(-10.0, std::min(10.0, weights[j]));
            }
            bias = std::max(-10.0, std::min(10.0, bias));
        }

        feature_weights_[step] = weights;
        bias_terms_[step] = bias;
    }

    is_trained_ = true;
}
};

struct CudaMemoryAlloc {
  uint64_t address;
  uint64_t size;
  double duration_ms;
  cudaError_t ret;
  std::chrono::steady_clock::time_point timestamp; // 添加时间戳
  uint64_t caller_func_off;                        // 调用者函数偏移地址

  CudaMemoryAlloc()
      : address(0), size(0), duration_ms(0), ret(cudaSuccess),
        timestamp(std::chrono::steady_clock::now()), caller_func_off(0) {}

  CudaMemoryAlloc(uint64_t addr, uint64_t sz, double dur, cudaError_t r,
                  std::chrono::steady_clock::time_point ts, uint64_t caller_off)
      : address(addr), size(sz), duration_ms(dur), ret(r), timestamp(ts),
        caller_func_off(caller_off) {}
};

class FragmentDetector {
public:
  FragmentDetector(Logger &logger, UprobeProfiler &profiler)
      : logger_(logger), profiler_(profiler) {
    // int deviceId = 0;
    // cudaGetDevice(&deviceId);
    // cudaDeviceGetDefaultMemPool(&pool, deviceId);
  }

  // void addEvent(const FragmentEvent& event);
  // bool analyzeCoalescing(uint64_t address, size_t size);
  // int detectStridePattern(uint64_t address);

  // void recordAccessPattern(uint64_t address, size_t size, uint64_t time);
  // void analyzeAccessPatterns();
  void detectCudaMemoryFragmentation(
      std::unordered_map<uint32_t, std::map<uint64_t, CudaMemoryAlloc>>
          memory_map);

private:
  void writeMemoryAllocationsToFile(
      std::vector<std::pair<uint64_t, uint64_t>> allocations, bool is_oom);
  FragmentationMetrics calculateCoreFragmentationMetrics(
      const std::vector<std::pair<uint64_t, uint64_t>> &allocations,
      uint64_t totalAllocatedSize);

  void calculateBasicGapMetrics(
      const std::vector<std::pair<uint64_t, uint64_t>> &allocations,
      uint64_t totalAllocatedSize, FragmentationMetrics &metrics);

  void calculateExternalFragmentationMetrics(FragmentationMetrics &metrics,
                                             uint64_t pageSize);

  void calculateAllocationPatternMetrics(
      const std::vector<std::pair<uint64_t, uint64_t>> &allocations,
      uint64_t totalAllocatedSize, FragmentationMetrics &metrics,
      uint64_t smallAllocThreshold);

  void calculateSpaceEfficiencyMetrics(FragmentationMetrics &metrics,
                                       uint64_t totalAllocatedSize);

  double calculateFinalFragmentationScore(const FragmentationMetrics &metrics);

  void outputFragmentationAnalysis(
      const FragmentationMetrics &metrics, double fragmentationScore,
      const std::vector<std::pair<uint64_t, uint64_t>> &allocations);

  double calculateTrendSlope(const std::deque<double> &data);
  // double calculatePredictionConfidence(double volatility, size_t
  // data_points); FragmentationPrediction calculateTrendBasedPrediction(const
  // FragmentationMetrics &current_metrics,
  //                               double current_score);
  // void applyCorrelationCorrection(FragmentationPrediction &prediction,
  //                                 const FragmentationMetrics
  //                                 &current_metrics);
  // void applyPatternBasedAdjustment(FragmentationPrediction &prediction,
  //                                  const FragmentationMetrics
  //                                  &current_metrics);
  // void
  // assessRiskAndGenerateWarnings(FragmentationPrediction &prediction,
  //                               const FragmentationMetrics &current_metrics);
  FragmentationPrediction
  predictFragmentationScore(const FragmentationMetrics &current_metrics,
                            double current_score);
  void assessRiskLevel(FragmentationPrediction &prediction,
                       double max_predicted_score);
  void generateWarnings(FragmentationPrediction &prediction,
                        const FragmentationMetrics &current_metrics,
                        double current_score);
  void outputPredictionAnalysis(const FragmentationPrediction &prediction,
                                double fragmentationScore);
  // std::vector<FragmentEvent> events_;
  // std::map<uint64_t, uint32_t> density_map_;
  // std::mutex mutex_;
  Logger logger_;
  UprobeProfiler &profiler_;
  // std::vector<AccessPattern> accessHistory_;
  cudaMemPool_t pool;
  HistoricalMetrics historical_metrics_;
  TimeSeriesPredictor time_series_predictor_;
};

} // namespace NeuTracer