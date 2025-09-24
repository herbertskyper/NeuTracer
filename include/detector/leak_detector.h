#ifndef NEUTRACER_LEAK_DETECTOR_H
#define NEUTRACER_LEAK_DETECTOR_H

#include <argp.h>
#include <bpf/libbpf.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// #include "snoop/gpu_snoop.h"
#include "utils/Logger.h"
#include "utils/SymUtils.h"
#include "utils/UprobeProfiler.h"

namespace NeuTracer {

// 泄漏分数结构
struct LeakScore {
  uint32_t frees;                                         // 释放次数
  uint32_t mallocs;                                       // 分配次数
  uint64_t total_size;                                    // 总分配大小
  uint64_t first_total_size;                              // 首次分配时的总大小
  std::chrono::steady_clock::time_point first_allocation; // 首次分配时间
  std::chrono::steady_clock::time_point last_allocation;  // 最后分配时间
  double leak_score;                                      // 泄漏分数（0-100%）
  double growth_rate;                                     // 内存增长率（0-1.0）
  std::string caller_func_name;                           // 调用者函数名称
  bool is_reported;                                       // 是否已报告过泄漏
  StackFrame frame;                                       // 调用者堆栈帧

  LeakScore()
      : frees(0), mallocs(0), total_size(0), first_total_size(0),
        leak_score(0.0), growth_rate(0.0), is_reported(0) {} // 初始化成员变量
};

// 泄漏检测器类
class LeakDetector {
private:
  // 最高水位标记
  uint64_t high_water_mark_;

  // 内存采样器阈值（字节）
  uint64_t memory_threshold_;
  uint64_t current_memory_usage_;

  // 互斥锁保护并发访问
  std::mutex detector_mutex_;
  Logger logger_;
  UprobeProfiler &profiler_;

public:
  LeakDetector(Logger &logger, UprobeProfiler &profiler,
                      uint64_t threshold = 10 * 1024 * 1024); // 默认10MB阈值

  // 核心算法函数
  void recordAllocation(uint64_t address, uint64_t size, uint64_t caller_offset,
                        uint64_t total_size, StackFrame frame,bool is_frame);
  void recordDeallocation(uint64_t address, uint64_t caller_offset,
                          uint64_t size);
  bool checkHighWaterMark(uint64_t new_memory_usage);
  void updateLeakScores(uint64_t last_obj_addr);
  double calculateLeakProbability(const LeakScore &score);
  double calculateMemoryGrowthRate(const LeakScore &score);
  std::vector<std::pair<StackFrame, double>> detectLeaks();
  void generateLeakReport();

  // 辅助函数
  void setMemoryThreshold(uint64_t threshold) { memory_threshold_ = threshold; }
  uint64_t getHighWaterMark() const { return high_water_mark_; }
  size_t getLeakScoreCount() const { return leak_scores_.size(); }

  // 泄漏分数映射：调用者偏移 -> 泄漏分数
  std::unordered_map<uint64_t, LeakScore> leak_scores_;
  // 泄漏检测配置
  static constexpr double LEAK_PROBABILITY_THRESHOLD = 0.95; // 95%
  static constexpr double MEMORY_GROWTH_THRESHOLD = 0.01;    // 1%
  uint64_t last_obj_addr_ = -1;
};

} // namespace NeuTracer

#endif // NEUTRACER_LEAK_DETECTOR_H