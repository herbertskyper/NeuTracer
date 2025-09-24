#include <argp.h>
#include <bpf/libbpf.h>
#include <chrono>

#include <cstdint>
#include <cstdio>
#include <iomanip>

#include <sstream>
#include <string>
#include <vector>
#include <numeric>

#include "detector/fragment_detector.h"
namespace NeuTracer {

void FragmentDetector::detectCudaMemoryFragmentation(
    std::unordered_map<uint32_t, std::map<uint64_t, CudaMemoryAlloc>>
        memory_map) {
  // 1. 收集所有已分配的内存块信息
  std::vector<std::pair<uint64_t, uint64_t>>
      allocations; // (start_address, size)
  bool is_oom = false;
  uint64_t totalAllocatedSize = 0;

  {
    auto now = std::chrono::steady_clock::now();
    const int OOM_TIME_THRESHOLD_SEC = 10;

    // 从所有进程收集内存分配信息
    for (const auto &[pid, memMap] : memory_map) {
      for (const auto &[addr, alloc] : memMap) {
        if (alloc.ret == cudaErrorMemoryAllocation) {
          auto seconds_since_fail =
              std::chrono::duration_cast<std::chrono::seconds>(now -
                                                               alloc.timestamp)
                  .count();
          if (seconds_since_fail <= OOM_TIME_THRESHOLD_SEC) {
            is_oom = true;
          }
        }
        if (alloc.ret != cudaSuccess) {
          logger_.warn("进程 {} 的 CUDA 分配失败: 地址=0x{:x}, 错误码={}", pid,
                       addr, alloc.ret);
          continue;
        }
        allocations.push_back({addr, alloc.size});
        totalAllocatedSize += alloc.size;
      }
    }
  }

  writeMemoryAllocationsToFile(allocations, is_oom);
  if (allocations.empty()) {
    logger_.info("没有检测到CUDA内存分配，跳过碎片化分析");
    return;
  }

  // 2. 按地址排序
  std::sort(allocations.begin(), allocations.end());

  // 3. 计算核心碎片化指标
  FragmentationMetrics metrics =
      calculateCoreFragmentationMetrics(allocations, totalAllocatedSize);

  // 4. 计算综合碎片化评分 (0-100)
  double fragmentationScore = calculateFinalFragmentationScore(metrics);

  // 5. 输出分析结果
  outputFragmentationAnalysis(metrics, fragmentationScore, allocations);

  FragmentationPrediction prediction =
      predictFragmentationScore(metrics, fragmentationScore);
  outputPredictionAnalysis(prediction, fragmentationScore);
}

void FragmentDetector::writeMemoryAllocationsToFile(
    std::vector<std::pair<uint64_t, uint64_t>> allocations, bool is_oom) {
  // 生成带时间戳的文件名
  auto now = std::chrono::system_clock::now();
  auto now_time_t = std::chrono::system_clock::to_time_t(now);

  // 写入简化的文件头
  std::stringstream header_time_ss;
  header_time_ss << std::put_time(std::localtime(&now_time_t),
                                  "%Y-%m-%d %H:%M:%S");

  profiler_.add_trace_data_gpu_memfragment("# CUDA Memory Allocations - {}",
                                           header_time_ss.str());

  // 按地址排序
  // std::sort(allocations.begin(), allocations.end());

  // 写入记录
  for (const auto &[addr, size] : allocations) {
    profiler_.add_trace_data_gpu_memfragment("0x{:x}, {} bytes", addr, size);
  }
  profiler_.add_trace_data_gpu_memfragment("is_oom:{}", is_oom);
  profiler_.add_trace_data_gpu_memfragment(std::string(50, '-'));
}

FragmentationMetrics FragmentDetector::calculateCoreFragmentationMetrics(
    const std::vector<std::pair<uint64_t, uint64_t>> &allocations,
    uint64_t totalAllocatedSize) {

  FragmentationMetrics metrics;
  const uint64_t GPU_PAGE_SIZE = 4096;                    // 4KB页大小
  const uint64_t SMALL_ALLOC_THRESHOLD = 4 * 1024 * 1024; // 4MB阈值

  // 1. 计算基础指标和间隙
  calculateBasicGapMetrics(allocations, totalAllocatedSize, metrics);

  // 2. 计算核心指标1: 外部碎片化
  calculateExternalFragmentationMetrics(metrics, GPU_PAGE_SIZE);

  // 3. 计算核心指标2: 分配模式
  calculateAllocationPatternMetrics(allocations, totalAllocatedSize, metrics,
                                    SMALL_ALLOC_THRESHOLD);

  // 4. 计算核心指标3: 空间效率
  calculateSpaceEfficiencyMetrics(metrics, totalAllocatedSize);

  return metrics;
}

void FragmentDetector::calculateBasicGapMetrics(
    const std::vector<std::pair<uint64_t, uint64_t>> &allocations,
    uint64_t totalAllocatedSize, FragmentationMetrics &metrics) {

  std::vector<uint64_t> gaps;
  std::vector<uint64_t> allocation_sizes;

  // 计算地址空间范围
  if (!allocations.empty()) {
    metrics.total_address_space = allocations.back().first +
                                  allocations.back().second -
                                  allocations.front().first;
  }

  // 计算间隙
  for (size_t i = 0; i < allocations.size(); i++) {
    allocation_sizes.push_back(allocations[i].second);

    if (i > 0) {
      uint64_t prevEnd = allocations[i - 1].first + allocations[i - 1].second;
      uint64_t currStart = allocations[i].first;

      if (currStart > prevEnd) {
        uint64_t gap = currStart - prevEnd;
        if (gap > 1024) { // 只计算大于1KB的间隙
          gaps.push_back(gap);
          metrics.total_gap_size += gap;
        }
      }
    }
  }

  metrics.gaps = gaps;
  metrics.allocation_sizes = allocation_sizes;
  metrics.gap_count = gaps.size();
  metrics.largest_contiguous_space =
      gaps.empty() ? 0 : *std::max_element(gaps.begin(), gaps.end());

  // 计算平均分配大小
  if (!allocation_sizes.empty()) {
    metrics.avg_allocation_size =
        static_cast<double>(totalAllocatedSize) / allocation_sizes.size();
  }
}

void FragmentDetector::calculateExternalFragmentationMetrics(
    FragmentationMetrics &metrics, uint64_t pageSize) {
  // 1. 传统外部碎片化率
  if (metrics.total_address_space > 0) {
    metrics.external_fragmentation_ratio =
        static_cast<double>(metrics.total_gap_size) /
        metrics.total_address_space;
  }

  // 2. 内核评估指标
  metrics.free_pages = metrics.total_gap_size / pageSize;
  if (metrics.total_gap_size % pageSize != 0) {
    metrics.free_pages++; // 向上取整
  }

  // 计算目标分配块大小 (基于平均分配大小)
  uint64_t target_block_size = 0;
  if (metrics.avg_allocation_size > 0) {
    // GPU倾向于2的幂次分配，找到最接近的2的幂
    target_block_size = 1ULL << static_cast<uint32_t>(
                            std::ceil(std::log2(metrics.avg_allocation_size)));
    // 最小块大小为GPU页面大小
    target_block_size = std::max(target_block_size * 2, pageSize);
  } else {
    target_block_size = pageSize;
  }

  for (uint64_t gap : metrics.gaps) {
    if (gap >= target_block_size) {
      // 每个间隙能提供的目标大小块数
      metrics.free_blocks_suitable += gap / target_block_size;
    }
  }

  // 3. 计算内核不可用指数
  // if (metrics.free_pages > 0) {
  //     int64_t numerator = static_cast<int64_t>(metrics.free_pages) -
  //                        static_cast<int64_t>((metrics.free_blocks_suitable
  //                        << target_order) * 1000ULL);
  //     metrics.kernel_unusable_index =
  //     static_cast<double>(std::max<int64_t>(0LL, numerator)) /
  //     metrics.free_pages; logger_.info("free_pages: {}, free_blocks_suitable:
  //     {}, target_order: {}", metrics.free_pages,
  //     metrics.free_blocks_suitable, target_order);
  // }
  if (metrics.free_pages > 0 && target_block_size > 0) {
    // 计算目标块大小相对于GPU页面的比例
    uint64_t blocks_per_page =
        std::max(static_cast<uint64_t>(1), target_block_size / pageSize);
    uint64_t theoretical_blocks = metrics.free_pages / blocks_per_page;
    // logger_.info("free_pages: {}, free_blocks_suitable: {},
    // target_block_size: {}KB, blocks_per_page: {}, theoretical_blocks: {}",
    //              metrics.free_pages, metrics.free_blocks_suitable,
    //              target_block_size / 1024, blocks_per_page,
    //              theoretical_blocks);

    if (theoretical_blocks > 0) {
      // GPU碎片化指数：实际可用块数与理论块数的差异比例
      double utilization_ratio =
          static_cast<double>(metrics.free_blocks_suitable) /
          theoretical_blocks;
      metrics.kernel_unusable_index = 1.0 - std::min(1.0, utilization_ratio);
      metrics.kernel_unusable_index = pow(metrics.kernel_unusable_index, 0.25);
    } else {
      metrics.kernel_unusable_index = 1.0;
    }
  } else {
    metrics.kernel_unusable_index = 0.0;
  }
  //  logger_.info("GPU内存分析: free_pages: {}, free_blocks_suitable: {},
  //  target_block_size: {}KB, gpu_frag_index: {:.4f}",
  //              metrics.free_pages, metrics.free_blocks_suitable,
  //              target_block_size / 1024, metrics.kernel_unusable_index);
}

void FragmentDetector::calculateAllocationPatternMetrics(
    const std::vector<std::pair<uint64_t, uint64_t>> &allocations,
    uint64_t totalAllocatedSize, FragmentationMetrics &metrics,
    uint64_t smallAllocThreshold) {

  if (metrics.allocation_sizes.empty())
    return;

  // 1. 计算小分配比例
  uint32_t small_alloc_count = 0;
  for (uint64_t size : metrics.allocation_sizes) {
    if (size <= smallAllocThreshold) {
      small_alloc_count++;
    }
  }
  metrics.small_allocation_ratio =
      static_cast<double>(small_alloc_count) / metrics.allocation_sizes.size();

  // 2. 计算分配大小方差 (标准化)
  double variance = 0.0;
  for (uint64_t size : metrics.allocation_sizes) {
    variance += std::pow(size - metrics.avg_allocation_size, 2);
  }
  double std_dev = std::sqrt(variance / metrics.allocation_sizes.size());

  // 标准化方差 (变异系数)
  if (metrics.avg_allocation_size > 0) {
    metrics.allocation_size_variance = std_dev / metrics.avg_allocation_size;
  }
}

void FragmentDetector::calculateSpaceEfficiencyMetrics(
    FragmentationMetrics &metrics, uint64_t totalAllocatedSize) {
  // 计算空间利用效率
  if (metrics.total_address_space > 0) {
    metrics.utilization_efficiency =
        static_cast<double>(totalAllocatedSize) / metrics.total_address_space;
  }
  double largeGapRatio = 0.0;
  if (!metrics.gaps.empty() && metrics.total_gap_size > 0) {
    // 计算大于平均间隙大小2倍的间隙比例
    uint64_t avgGapSize = metrics.total_gap_size / metrics.gaps.size();
    uint64_t largeGapSize = 0;
    for (uint64_t gap : metrics.gaps) {
      if (gap > avgGapSize * 2) {
        largeGapSize += gap;
      }
    }
    largeGapRatio = static_cast<double>(largeGapSize) / metrics.total_gap_size;
  }
  metrics.large_gap_ratio = largeGapRatio;
}

double FragmentDetector::calculateFinalFragmentationScore(
    const FragmentationMetrics &metrics) {
  double score = 0.0;

  // 权重1: 外部碎片化 (50%) - 最重要的指标
  score += std::min(metrics.external_fragmentation_ratio*2,1.0) * 50.0;

  // 权重2: 内核不可用指数 (15%)
  double normalized_kernel_index =
      std::min(1.0, metrics.kernel_unusable_index / 10.0); // 归一化
  score += normalized_kernel_index * 15.0;

  // 权重3: 分配模式不良 (10%)
  // 小分配过多 + 分配大小差异过大
  double pattern_penalty =
      (metrics.small_allocation_ratio * 0.6) +
      (std::min(1.0, metrics.allocation_size_variance) * 0.4);
  score += pattern_penalty * 10.0;

  // 权重4: 空间利用效率低 (25%)
  score += metrics.large_gap_ratio * 25.0;

  // 确保评分在0-100之间
  return std::min(100.0, std::max(0.0, score));
}

// FragmentationPrediction FragmentDetector::predictFragmentationScore(
//     const FragmentationMetrics &current_metrics, double current_score) {

//   // 更新历史数据
//   historical_metrics_.addRecord(current_metrics, current_score);

//   FragmentationPrediction prediction;

//   if (!historical_metrics_.hasEnoughData()) {
//     // 数据不足，返回保守预测
//     prediction.predicted_score = current_score;
//     prediction.confidence = 0.3;
//     prediction.risk_level = "INSUFFICIENT_DATA";
//     prediction.warnings.push_back("历史数据不足，预测准确性有限");
//     return prediction;
//   }

//   // 1. 基于加权移动平均的趋势预测
//   prediction = calculateTrendBasedPrediction(current_metrics, current_score);

//   // 2. 基于指标关联性的修正
//   applyCorrelationCorrection(prediction, current_metrics);

//   // 3. 基于模式识别的调整
//   applyPatternBasedAdjustment(prediction, current_metrics);

//   // 4. 风险评估和预警
//   assessRiskAndGenerateWarnings(prediction, current_metrics);

//   return prediction;
// }

// FragmentationPrediction FragmentDetector::calculateTrendBasedPrediction(
//     const FragmentationMetrics &current_metrics, double current_score) {

//   FragmentationPrediction prediction;
//   const auto &scores = historical_metrics_.scores;

//   // 1. 计算加权移动平均 (最近的数据权重更高)
//   double weighted_avg = 0.0;
//   double weight_sum = 0.0;

//   for (size_t i = 0; i < scores.size(); i++) {
//     double weight = std::pow(1.1, i); // 指数权重
//     weighted_avg += scores[i] * weight;
//     weight_sum += weight;
//   }
//   weighted_avg /= weight_sum;

//   // 2. 计算趋势斜率 (线性回归)
//   double trend_slope = calculateTrendSlope(scores);
//   prediction.trend_factor =
//       std::max(-1.0, std::min(1.0, trend_slope / 10.0)); // 归一化

//   // 3. 计算波动性
//   double variance = 0.0;
//   for (double score : scores) {
//     variance += std::pow(score - weighted_avg, 2);
//   }
//   prediction.volatility = std::sqrt(variance / scores.size()) / 100.0; // 归一化

//   // 4. 基础预测分数
//   double trend_component = current_score + (trend_slope * 10.0); // 预测未来10步
//   double stability_component = weighted_avg * 0.3 + current_score * 0.7;

//   // 综合预测
//   prediction.predicted_score =
//       trend_component * 0.6 + stability_component * 0.4;
//   prediction.predicted_score =
//       std::max(0.0, std::min(100.0, prediction.predicted_score));

//   // 5. 计算置信度
//   prediction.confidence =
//       calculatePredictionConfidence(prediction.volatility, scores.size());

//   return prediction;
// }

// void FragmentDetector::applyCorrelationCorrection(
//     FragmentationPrediction &prediction,
//     const FragmentationMetrics &current_metrics) {

//   // 基于各指标的相关性进行修正
//   double correction = 0.0;

//   // 1. 外部碎片化率的影响
//   if (current_metrics.external_fragmentation_ratio > 0.5) {
//     double frag_trend =
//         calculateTrendSlope(historical_metrics_.external_frag_ratios);
//     if (frag_trend > 0.02) { // 外部碎片化快速上升
//       correction += 5.0;
//       prediction.warnings.push_back("外部碎片化率快速上升");
//     }
//   }

//   // 2. 内核不可用指数的影响
//   if (current_metrics.kernel_unusable_index > 0.3) {
//     double kernel_trend =
//         calculateTrendSlope(historical_metrics_.kernel_indices);
//     if (kernel_trend > 0.05) {
//       correction += 8.0;
//       prediction.warnings.push_back("内存可用性持续恶化");
//     }
//   }

//   // 3. 小分配比例的影响
//   if (current_metrics.small_allocation_ratio > 0.6) {
//     double small_alloc_trend =
//         calculateTrendSlope(historical_metrics_.small_alloc_ratios);
//     if (small_alloc_trend > 0.03) {
//       correction += 3.0;
//       prediction.warnings.push_back("小分配比例持续增加");
//     }
//   }

//   // 4. 分配数量激增的影响
//   if (historical_metrics_.allocation_counts.size() >= 2) {
//     size_t recent_count = historical_metrics_.allocation_counts.back();
//     size_t prev_count =
//         historical_metrics_
//             .allocation_counts[historical_metrics_.allocation_counts.size() -
//                                2];

//     if (recent_count > prev_count * 1.5) { // 分配数量激增50%以上
//       correction += 6.0;
//       prediction.warnings.push_back("内存分配频率异常增高");
//     }
//   }

//   // 应用修正
//   prediction.predicted_score += correction;
//   prediction.predicted_score =
//       std::max(0.0, std::min(100.0, prediction.predicted_score));

//   // 修正后调整置信度
//   if (correction > 0) {
//     prediction.confidence *= 0.9; // 有额外风险因素时降低置信度
//   }
// }

// void FragmentDetector::applyPatternBasedAdjustment(
//     FragmentationPrediction &prediction,
//     const FragmentationMetrics &current_metrics) {

//   const auto &scores = historical_metrics_.scores;

//   // 1. 检测锯齿波动模式 (频繁上下波动)
//   if (scores.size() >= 4) {
//     int direction_changes = 0;
//     for (size_t i = 2; i < scores.size(); i++) {
//       bool prev_increasing = scores[i - 1] > scores[i - 2];
//       bool curr_increasing = scores[i] > scores[i - 1];
//       if (prev_increasing != curr_increasing) {
//         direction_changes++;
//       }
//     }

//     if (direction_changes >= 2) {        // 频繁波动
//       prediction.predicted_score += 2.0; // 波动通常预示着不稳定
//       prediction.warnings.push_back("检测到碎片化分数频繁波动");
//       prediction.confidence *= 0.85;
//     }
//   }

//   // 2. 检测阶跃跳跃模式 (突然大幅变化)
//   if (scores.size() >= 2) {
//     double last_change = std::abs(scores.back() - scores[scores.size() - 2]);
//     if (last_change > 15.0) {                          // 突然变化超过15分
//       prediction.predicted_score += last_change * 0.3; // 延续变化趋势
//       prediction.warnings.push_back("检测到碎片化分数突然大幅变化");
//       prediction.confidence *= 0.8;
//     }
//   }

//   // 3. 检测持续恶化模式
//   if (scores.size() >= 3) {
//     bool consistently_increasing = true;
//     for (size_t i = 1; i < scores.size(); i++) {
//       if (scores[i] <= scores[i - 1]) {
//         consistently_increasing = false;
//         break;
//       }
//     }

//     if (consistently_increasing && scores.back() > scores.front() + 10) {
//       prediction.predicted_score += 5.0; // 持续恶化的惯性
//       prediction.warnings.push_back("检测到碎片化持续恶化趋势");
//     }
//   }

//   // 确保预测分数在合理范围内
//   prediction.predicted_score =
//       std::max(0.0, std::min(100.0, prediction.predicted_score));
// }

// void FragmentDetector::assessRiskAndGenerateWarnings(
//     FragmentationPrediction &prediction,
//     const FragmentationMetrics &current_metrics) {

//   // 风险等级评估
//   if (prediction.predicted_score >= 80) {
//     prediction.risk_level = "CRITICAL";
//   } else if (prediction.predicted_score >= 70) {
//     prediction.risk_level = "HIGH";
//   } else if (prediction.predicted_score >= 50) {
//     prediction.risk_level = "MEDIUM";
//   } else if (prediction.predicted_score >= 30) {
//     prediction.risk_level = "LOW";
//   } else {
//     prediction.risk_level = "MINIMAL";
//   }

//   // 生成具体预警
//   if (prediction.trend_factor > 0.3) {
//     prediction.warnings.push_back("碎片化呈上升趋势，建议提前干预");
//   }

//   if (prediction.volatility > 0.2) {
//     prediction.warnings.push_back("碎片化波动较大，系统不稳定");
//   }

//   if (prediction.confidence < 0.6) {
//     prediction.warnings.push_back("预测置信度较低，建议增加监控频率");
//   }

//   // 基于当前指标的风险预警
//   if (current_metrics.external_fragmentation_ratio > 0.4 &&
//       prediction.trend_factor > 0.1) {
//     prediction.warnings.push_back("外部碎片化已达警戒线且仍在上升");
//   }

//   if (current_metrics.kernel_unusable_index > 0.5) {
//     prediction.warnings.push_back("内存可用性严重下降，可能影响大块分配");
//   }
// }

// 辅助函数
// double FragmentDetector::calculateTrendSlope(const std::deque<double> &data) {
//   if (data.size() < 2)
//     return 0.0;

//   // 线性回归计算斜率
//   double n = data.size();
//   double sum_x = n * (n - 1) / 2; // 0+1+2+...+(n-1)
//   double sum_y = std::accumulate(data.begin(), data.end(), 0.0);
//   double sum_xy = 0.0;
//   double sum_x2 = n * (n - 1) * (2 * n - 1) / 6; // 0²+1²+2²+...+(n-1)²

//   for (size_t i = 0; i < data.size(); i++) {
//     sum_xy += i * data[i];
//   }

//   double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
//   return slope;
// }

// double FragmentDetector::calculatePredictionConfidence(double volatility,
//                                                        size_t data_points) {
//   // 基于波动性和数据点数量计算置信度
//   double volatility_penalty = std::max(0.0, 1.0 - volatility * 3.0);
//   double data_bonus = std::min(1.0, data_points / 10.0);
//   return (volatility_penalty * 0.7 + data_bonus * 0.3);
// }

void FragmentDetector::outputFragmentationAnalysis(
    const FragmentationMetrics &metrics, double fragmentationScore,
    const std::vector<std::pair<uint64_t, uint64_t>> &allocations) {

  logger_.info("=== CUDA内存碎片化分析 (核心指标评估) ===");

  // 基础统计
  uint64_t totalAllocated = std::accumulate(
      allocations.begin(), allocations.end(), 0ULL,
      [](uint64_t sum, const auto &alloc) { return sum + alloc.second; });

  logger_.info("基础统计:");
  logger_.info("  - 总分配块数: {}", allocations.size());
  logger_.info("  - 总分配内存: {:.2f}MB", totalAllocated / (1024.0 * 1024.0));
  logger_.info("  - 平均分配大小: {:.2f}KB",
               metrics.avg_allocation_size / 1024.0);
  logger_.info("  - 检测间隙数: {}", metrics.gap_count);
  logger_.info("  - 总间隙大小: {:.2f}MB",
               metrics.total_gap_size / (1024.0 * 1024.0));

  // 核心碎片化指标
  logger_.info("核心碎片化指标:");
  logger_.info("  - 外部碎片化率: {:.2f}%",
               metrics.external_fragmentation_ratio * 100);
  logger_.info("  - 内核不可用指数: {:.6f}", metrics.kernel_unusable_index);
  logger_.info("  - 可用页数: {}", metrics.free_pages);
  logger_.info("  - 适合分配的空闲块数: {}", metrics.free_blocks_suitable);
  logger_.info("  - 小分配(<4MB)比例: {:.1f}%",
               metrics.small_allocation_ratio * 100);
  logger_.info("  - 分配大小变异系数: {:.3f}",
               metrics.allocation_size_variance);
  logger_.info("  - 空间利用效率: {:.2f}%",
               metrics.utilization_efficiency * 100);
  logger_.info("  - 最大连续空间: {:.2f}MB",
               metrics.largest_contiguous_space / (1024.0 * 1024.0));
  logger_.info("  - 大间隙比例: {:.2f}%", metrics.large_gap_ratio * 100);


  // 综合评分
  logger_.info("综合碎片化评分: {:.1f}/100", fragmentationScore);

  // 根据评分给出建议
  if (fragmentationScore >= 80) {
    logger_.error("检测到严重内存碎片化 (评分: {:.1f})", fragmentationScore);

    // 具体问题诊断
    if (metrics.external_fragmentation_ratio > 0.4) {
      logger_.info("  [关键] 外部碎片化严重({:.1f}%)，需要内存整理",
                   metrics.external_fragmentation_ratio * 100);
    }
    if (metrics.kernel_unusable_index > 5.0) {
      logger_.info("  [关键] 内核评估显示可用内存利用率极低({:.2f})",
                   metrics.kernel_unusable_index);
    }
    if (metrics.small_allocation_ratio > 0.7) {
      logger_.info("  [警告] 小分配过多({:.1f}%)，建议合并分配",
                   metrics.small_allocation_ratio * 100);
    }
    if (metrics.utilization_efficiency < 0.6) {
      logger_.info("  [警告] 空间利用效率低({:.1f}%)，存在大量浪费",
                   metrics.utilization_efficiency * 100);
    }

    logger_.info("建议措施:");
    logger_.info("  1. 立即执行内存整理或重启CUDA上下文");
    logger_.info("  2. 重新设计内存分配策略，减少小块分配");
    logger_.info("  3. 使用内存池管理器统一管理分配");

  } else if (fragmentationScore >= 60) {
    logger_.warn("检测到中度内存碎片化 (评分: {:.1f})", fragmentationScore);
    logger_.info("建议监控分配模式，考虑预防性优化");

  } else if (fragmentationScore >= 40) {
    logger_.info("轻度内存碎片化 (评分: {:.1f})", fragmentationScore);
    logger_.info("内存管理状况良好，建议持续监控");

  } else {
    logger_.info("内存碎片化程度很低 (评分: {:.1f})", fragmentationScore);
    logger_.info("内存管理非常高效");
  }

  // 如果碎片化严重且分配数量不多，输出详细信息
  //   if (fragmentationScore >= 70 && allocations.size() <= 50) {
  //     logger_.info("内存分配详情 (按地址排序):");
  //     for (size_t i = 0; i < std::min(allocations.size(), size_t(10)); i++) {
  //       uint64_t addr = allocations[i].first;
  //       uint64_t size = allocations[i].second;

  //       uint64_t gap = 0;
  //       if (i > 0) {
  //         uint64_t prevEnd = allocations[i - 1].first + allocations[i -
  //         1].second; gap = addr > prevEnd ? addr - prevEnd : 0;
  //       }

  //       logger_.info("  块 {}: 地址=0x{:x}, 大小={}KB, 前置间隙={}KB", i,
  //       addr,
  //                    size / 1024, gap / 1024);
  //     }
  //     if (allocations.size() > 10) {
  //       logger_.info("  ... 省略其余 {} 个分配块", allocations.size() - 10);
  //     }
  //   }
}

FragmentationPrediction FragmentDetector::predictFragmentationScore(
    const FragmentationMetrics& current_metrics, double current_score) {
    
    FragmentationPrediction prediction;
    
    // 计算趋势和波动性 (简化版本)
    double trend_slope = 0.0;
    double volatility = 0.0;
    // 使用简单的历史分数来计算趋势
    if (historical_metrics_.scores.size() >= 2) {
     
        trend_slope = calculateTrendSlope(historical_metrics_.scores);
        // 计算波动性
        if (historical_metrics_.scores.size() >= 3) {
            double mean_score = std::accumulate(historical_metrics_.scores.begin(), 
                                              historical_metrics_.scores.end(), 0.0) / 
                                              historical_metrics_.scores.size();
            double variance = 0.0;
            for (double score : historical_metrics_.scores) {
                variance += std::pow(score - mean_score, 2);
            }
            volatility = std::sqrt(variance / historical_metrics_.scores.size()) / 100.0;
        }
    }
    historical_metrics_.addRecord(current_score);
    time_series_predictor_.addSample(current_metrics, current_score);
    
    // 检查是否有足够数据进行预测
    if (!time_series_predictor_.hasEnoughData()) {
        prediction.predicted_scores = {current_score, current_score, current_score};
        prediction.time_labels = {"下一次检测", "2次检测后", "3次检测后"};
        prediction.confidence = 0.3;
        prediction.risk_level = "INSUFFICIENT_DATA";
        prediction.warnings.push_back("历史数据不足，预测准确性有限");
        prediction.trend_factor = trend_slope;
        prediction.volatility = volatility;
        return prediction;
    }
    
    // 使用时间序列预测器进行预测
    prediction.predicted_scores = time_series_predictor_.predictFutureScores(current_metrics);
    prediction.time_labels = {"下一次检测", "2次检测后", "3次检测后"};
    
    // 检查预测是否成功
    bool prediction_failed = true;
    for (double score : prediction.predicted_scores) {
        if (score >= 0) {
            prediction_failed = false;
            break;
        }
    }
    
    if (prediction_failed) {
        prediction.predicted_scores = {current_score, current_score, current_score};
        prediction.confidence = 0.3;
    } else {
        prediction.confidence = time_series_predictor_.getConfidence();
    }
    
    // 设置趋势因子和波动性
    prediction.trend_factor = trend_slope;
    // logger_.info("trend_slope after: {}", trend_slope);
    prediction.volatility = volatility;
    
    // 风险等级评估（基于预测的最高分数）
    double max_predicted_score = *std::max_element(prediction.predicted_scores.begin(), 
                                                  prediction.predicted_scores.end());
    assessRiskLevel(prediction,max_predicted_score);
    
    // 生成预警信息
    generateWarnings(prediction, current_metrics, current_score);
    
    return prediction;
}

// 简化的风险评估
void FragmentDetector::assessRiskLevel(FragmentationPrediction& prediction,double max_predicted_score) {
    if (max_predicted_score >= 80) {
        prediction.risk_level = "CRITICAL";
    } else if (max_predicted_score >= 70) {
        prediction.risk_level = "HIGH";
    } else if (max_predicted_score >= 50) {
        prediction.risk_level = "MEDIUM";
    } else if (max_predicted_score >= 30) {
        prediction.risk_level = "LOW";
    } else {
        prediction.risk_level = "MINIMAL";
    }
}

// 简化的预警生成
void FragmentDetector::generateWarnings(FragmentationPrediction& prediction, 
                                                  const FragmentationMetrics& current_metrics,
                                                  double current_score) {
    // 检查预测趋势
    bool is_worsening = false;
    for (size_t i = 0; i < prediction.predicted_scores.size(); i++) {
        if (prediction.predicted_scores[i] > current_score + 5.0) {
            is_worsening = true;
            prediction.warnings.push_back("预测" + prediction.time_labels[i] + 
                                         "碎片化将显著恶化至" + 
                                         std::to_string(prediction.predicted_scores[i]) + "分");
        }
    }
    
    // 检查急剧恶化
    for (size_t i = 1; i < prediction.predicted_scores.size(); i++) {
        if (prediction.predicted_scores[i] - prediction.predicted_scores[i-1] > 10.0) {
            prediction.warnings.push_back("预测在" + prediction.time_labels[i] + 
                                         "将出现急剧恶化");
        }
    }
    
    // 基于趋势的预警
    if (prediction.trend_factor > 0.3) {
        prediction.warnings.push_back("碎片化呈明显上升趋势，建议立即采取预防措施");
    }
    
    // 基于波动性的预警
    if (prediction.volatility > 0.2) {
        prediction.warnings.push_back("碎片化波动较大，系统状态不稳定");
    }
    
    // 基于置信度的预警
    if (prediction.confidence < 0.6) {
        prediction.warnings.push_back("预测置信度较低，建议增加监控频率");
    }
    
    // 基于当前指标的预警
    if (current_metrics.external_fragmentation_ratio > 0.4 && prediction.trend_factor > 0.1) {
        prediction.warnings.push_back("外部碎片化已达警戒线且仍在上升");
    }
    
    if (current_metrics.kernel_unusable_index > 0.5) {
        prediction.warnings.push_back("内存可用性严重下降，可能影响大块分配");
    }
}

// 保留原有的辅助函数
double FragmentDetector::calculateTrendSlope(const std::deque<double>& data) {
    if (data.size() < 2) return 0.0;
    
    // 简单线性回归计算斜率
    double n = data.size();
    double sum_x = n * (n - 1) / 2; // 0+1+2+...+(n-1)
    double sum_y = std::accumulate(data.begin(), data.end(), 0.0);
    double sum_xy = 0.0;
    double sum_x2 = n * (n - 1) * (2 * n - 1) / 6; // 0²+1²+2²+...+(n-1)²
    
    for (size_t i = 0; i < data.size(); i++) {
        sum_xy += i * data[i];
    }
    
    double denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-10) return 0.0;  // 避免除零
    
    double slope = (n * sum_xy - sum_x * sum_y) / denominator;
    // logger_.info("trend_slope before: {}", slope);
    return slope;
}

void FragmentDetector::outputPredictionAnalysis(
    const FragmentationPrediction& prediction, double fragmentationScore) {
    
    logger_.info("=== 碎片化趋势预测分析（时间序列预测）===");
    logger_.info("当前碎片化分数: {:.1f}/100", fragmentationScore);
    
    // 输出未来预测
    logger_.info("未来碎片化分数预测 (置信度: {:.1f}%):", prediction.confidence * 100);
    for (size_t i = 0; i < prediction.predicted_scores.size(); i++) {
        logger_.info("  - {}: {:.1f}/100", 
                    prediction.time_labels[i], prediction.predicted_scores[i]);
    }
    
    logger_.info("整体趋势因子: {:.3f} ({})", prediction.trend_factor,
                prediction.trend_factor > 0 ? "上升趋势" : "下降趋势");
    logger_.info("波动性: {:.3f}", prediction.volatility);
    logger_.info("风险等级: {}", prediction.risk_level);
    
    if (!prediction.warnings.empty()) {
        logger_.warn("预测预警:");
        for (const auto& warning : prediction.warnings) {
            logger_.warn("  - {}", warning);
        }
    }
    
    // 根据预测给出具体建议
    double max_predicted = *std::max_element(prediction.predicted_scores.begin(), 
                                           prediction.predicted_scores.end());
    
    if (max_predicted > fragmentationScore + 10) {
        logger_.warn("预测显示碎片化将在未来显著恶化，建议:");
        logger_.info("  1. 立即执行预防性内存整理");
        logger_.info("  2. 增加监控频率到每次检测");
        logger_.info("  3. 准备内存分配策略调整方案");
        logger_.info("  4. 考虑在预测恶化前重启应用或CUDA上下文");
    } else if (max_predicted > fragmentationScore + 5) {
        logger_.warn("预测显示碎片化可能适度恶化，建议:");
        logger_.info("  1. 密切监控内存分配模式");
        logger_.info("  2. 准备内存优化措施");
    }
}

} // namespace NeuTracer