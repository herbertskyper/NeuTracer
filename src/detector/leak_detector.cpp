

#include <argp.h>
#include <bpf/libbpf.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fmt/core.h>
#include <string>
#include <vector>

#include "detector/leak_detector.h"
#include "utils/Logger.h"
#include "utils/SymUtils.h"
#include "utils/UprobeProfiler.h"

namespace NeuTracer {

LeakDetector::LeakDetector(Logger &logger,
                                         UprobeProfiler &profiler,
                                         uint64_t threshold)
    : high_water_mark_(0), memory_threshold_(threshold),
      current_memory_usage_(0), logger_(logger), profiler_(profiler) {}

// 1. 记录分配（包含最高水位标记检测和采样分配记录）
void LeakDetector::recordAllocation(uint64_t address, uint64_t size,
                                           uint64_t caller_offset,
                                           uint64_t total_size,
                                           StackFrame frame,bool is_frame) {
  std::lock_guard<std::mutex> lock(detector_mutex_);

  current_memory_usage_ += size;

  // 最高水位标记检测
  if (checkHighWaterMark(current_memory_usage_)) {
    // 达到新的最高水位标记
    if (last_obj_addr_ == -1) {
      last_obj_addr_ = caller_offset;
      return;
    }
    // 更新之前采样对象的泄漏分数
    updateLeakScores(last_obj_addr_);

    // 更新该分配位置的mallocs计数
    auto &score = leak_scores_[caller_offset];
    if (score.first_total_size == 0) {
      score.first_total_size = total_size; // 初始化第一次分配的大小
      score.frame = frame;                 // 记录调用函数名
      score.first_allocation =
          std::chrono::steady_clock::now(); // 记录第一次分配时间
    }
    score.mallocs++;
    score.total_size += size;
    // if(score.total_size < 10446744071968915456ULL) {
    // profiler_.add_trace_data_gpu_memleak(
    //     "记录分配：地址=0x{:x}, 大小={} 字节, 调用函数={},
    //     当前内存使用={}:{}", caller_offset, size, caller_func_name,
    //     score.total_size,current_memory_usage_);
    score.last_allocation = std::chrono::steady_clock::now();
    last_obj_addr_ = caller_offset;
    if(is_frame)
        score.frame = frame; // 更新调用函数名
    // logger_.info("记录分配：地址=0x{:x}, 大小={} 字节, 调用函数={},
    // 当前内存使用={}",
    //          caller_offset, score.total_size, caller_func_name,
    //          current_memory_usage_);
  } else {
    auto it = leak_scores_.find(caller_offset);

    if (it != leak_scores_.end()) {
      if(is_frame)
        it->second.frame = frame; 
      it->second.mallocs++;
      it->second.total_size += size;
      it->second.last_allocation = std::chrono::steady_clock::now();
    }
  }
}

// 2. 记录释放（包含回收检查）
void LeakDetector::recordDeallocation(uint64_t address,
                                             uint64_t caller_offset,
                                             uint64_t size) {
  std::lock_guard<std::mutex> lock(detector_mutex_);

  current_memory_usage_ -= size;

  auto it = leak_scores_.find(caller_offset);
  if (it != leak_scores_.end()) {
    it->second.frees++;
    if (it->second.total_size > 0) {
      it->second.total_size -= size;
    }
    // ! 会导致无符号数小于0的情况
    // it->second.total_size -= size;
  }
}

// 3. 最高水位标记检测
bool LeakDetector::checkHighWaterMark(uint64_t new_memory_usage) {
  if (new_memory_usage > high_water_mark_ + memory_threshold_) {
    high_water_mark_ = new_memory_usage;
    return true;
  }
  return false;
}

// 4. 泄漏分数更新
void LeakDetector::updateLeakScores(uint64_t last_obj_addr) {
  // 在达到新的最大内存占用量时更新分数
  // 这个函数在recordAllocation中已经被调用，主要用于清理和统计

  uint64_t caller_offset = last_obj_addr;
  auto it = leak_scores_.find(caller_offset);
  if (it != leak_scores_.end()) {
    if (it->second.is_reported)
      return;
    // 如果对象未被回收，mallocs - frees 会保持差值
    // 分数已在recordAllocation和recordDeallocation中更新
    it->second.leak_score = calculateLeakProbability(it->second);
    it->second.growth_rate = calculateMemoryGrowthRate(it->second);
    // logger_.info(
    //     "更新泄漏分数：地址=0x{:x}, 分配大小={} 字节, "
    //     "泄漏概率={:.2f}, 内存增长率={:.2f}",
    //     caller_offset, it->second.total_size, it->second.leak_score,
    //     it->second.growth_rate);
    if (it->second.leak_score > LEAK_PROBABILITY_THRESHOLD &&
        it->second.growth_rate > MEMORY_GROWTH_THRESHOLD) {
      logger_.warn(
          "可能的内存泄漏检测到: 总分配大小={} bytes, "
          "泄漏概率={:.2f}, 内存增长率={:.2f}, 现在内存总使用量={} bytes",
          it->second.total_size, it->second.leak_score, it->second.growth_rate,
          current_memory_usage_);
      it->second.is_reported = true;
    }
  }
}

// 5. 泄漏可能性计算（拉普拉斯继承规则）
double LeakDetector::calculateLeakProbability(const LeakScore &score) {
  // 使用拉普拉斯继承规则计算泄漏概率
  // 公式: 1.0 - (frees + 1) / (mallocs - frees + 2)

  if (score.mallocs == 0) {
    return 0.0; // 没有分配，不可能泄漏
  }

  uint32_t unfreed = score.mallocs - score.frees;
  double probability =
      1.0 - static_cast<double>(score.frees + 1) / (unfreed + 2);

  return std::max(0.0, std::min(1.0, probability)); // 确保在[0,1]范围内
}

double LeakDetector::calculateMemoryGrowthRate(const LeakScore &score) {
  // 使用内存增长率公式计算内存增长率
  // 公式: (当前内存使用 - 最初内存使用) / 最初内存使用
  // logger_.info(
  //     "计算内存增长率：分配次数={}, 总大小={} 字节, "
  //     "第一次分配时间={}, 最后分配时间={}",
  //     score.mallocs, score.total_size,
  //     score.first_allocation.time_since_epoch().count(),
  //     score.last_allocation.time_since_epoch().count());
  if (score.first_allocation == score.last_allocation) {
    return 0.0; // 没有分配，无法计算增长率
  }
  uint64_t memory_growth = score.total_size;
  return static_cast<double>(memory_growth) /
         (score.first_total_size + score.total_size);
}

// 6. 泄漏检测
std::vector<std::pair<StackFrame, double>> LeakDetector::detectLeaks() {
  std::lock_guard<std::mutex> lock(detector_mutex_);

  std::vector<std::pair<StackFrame, double>> potential_leaks;

  for (const auto &[caller_offset, score] : leak_scores_) {

    double leak_score = score.leak_score;
    double growth_rate = score.growth_rate;
    // logger_.info("检测泄漏：caller_offset:0x{:x}, mallocs:{}, frees:{},
    // total_size:{}, "
    //             "first_allocation:{}, last_allocation:{}, leak_score:{},
    //             growth_rate:{}", caller_offset, score.mallocs, score.frees,
    //             score.total_size,
    //             score.first_allocation.time_since_epoch().count(),
    //             score.last_allocation.time_since_epoch().count(), leak_score,
    //             growth_rate);
    if (leak_score < LEAK_PROBABILITY_THRESHOLD ||
        growth_rate < MEMORY_GROWTH_THRESHOLD)
      continue;
    // logger_.info("caller_offset:0x{:x},leak_score:{},growth_rate:{}",caller_offset,
    // leak_score, growth_rate);

    double leak_probability = 0.0;
    if (score.mallocs > 0) {
      // 计算时间差（转换为秒）
      auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                           score.last_allocation - score.first_allocation)
                           .count();
      // fmt::print("avg_alloc_size:{},time_diff:{}", 0, time_diff);
      if (time_diff > 0) {
        // 计算泄漏率：平均分配大小 / 时间差
        double avg_alloc_size = static_cast<double>(score.total_size) /
                                (1024 * 1024) / score.mallocs;
        // fmt::print("avg_alloc_size:{},time_diff:{}", avg_alloc_size,
        // time_diff);
        leak_probability = avg_alloc_size / time_diff; // bytes per second
      }
    }

    // 过滤：仅报告泄漏可能性超过95%阈值的泄漏
    if (leak_score >= LEAK_PROBABILITY_THRESHOLD &&
        growth_rate >= MEMORY_GROWTH_THRESHOLD) {
      potential_leaks.emplace_back(score.frame, leak_probability);
    }
  }

  // 优先级排序：按泄漏概率降序排列
  std::sort(potential_leaks.begin(), potential_leaks.end(),
            [](const auto &a, const auto &b) {
              return a.second > b.second; // 按概率降序
            });

  return potential_leaks;
}

// 7. 生成泄漏报告
void LeakDetector::generateLeakReport() {
  auto potential_leaks = detectLeaks();

  if (potential_leaks.empty()) {
    profiler_.add_trace_data_gpu_memleak("没有检测到内存泄漏");
    logger_.info("没有检测到内存泄漏");
    return; // 没有检测到泄漏
  } else {
    logger_.warn("检测到 {} 个潜在泄漏:", potential_leaks.size());

    for (size_t i = 0; i < potential_leaks.size(); i++) {
      StackFrame frame = potential_leaks[i].first;
      double probability = potential_leaks[i].second;

      profiler_.add_trace_data_gpu_memleak(
          "泄漏 #{}: 调用函数: {:016x}: {} @ 0x{:x}+0x{:x}, 泄漏率={:.1f}",
          i + 1, frame.address, frame.name.c_str(), frame.address, frame.offset,
          probability);
    }
  }
}

} // namespace NeuTracer
