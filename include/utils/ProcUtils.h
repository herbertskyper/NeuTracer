// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// #include <fmt/core.h>
#include <cstdio>

#include <string>
#include <vector>
#include "utils/Logger.h"
#include "utils/UprobeProfiler.h"

namespace NeuTracer {

#define PATH_MAX 4096

struct MemoryMapping {
  uintptr_t startAddr;
  uintptr_t endAddr;
  unsigned long long fileOffset;
  bool readable;
  bool writable;
  bool executable;
  bool shared;
  dev_t devMajor;
  dev_t devMinor;
  ino_t inode;
  std::string name;
};

class ProcUtils {
 public:
  std::vector<MemoryMapping> getAllMemoryMappings(pid_t pid);
  ProcUtils(Logger &logger, UprobeProfiler &profiler);
 private:
  Logger logger_;
  UprobeProfiler& profiler_;
};

} // namespace NeuTracer
