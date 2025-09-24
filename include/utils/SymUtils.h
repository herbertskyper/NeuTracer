// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdio>
// #include <filesystem>
#include <map>
// #include <set>
#include <string>
#include <vector>
#ifdef FBCODE_STROBELIGHT
#include "blazesym/blazesym.h" // @manual=fbsource//third-party/rust:blazesym-c-cxx
#else
#include "blazesym.h" // @manual=fbsource//third-party/rust:blazesym-c-cxx
#endif
#include "utils/Logger.h"
#include "utils/ProcUtils.h"
#include "utils/UprobeProfiler.h"
#include "config.h"

namespace NeuTracer {
struct StackFrame {
  std::string name;
  size_t address;
  std::string module;
  std::string file;
  size_t line;
  size_t offset;
  bool inlines;
};

struct SymbolInfo {
  std::string name;
  std::vector<std::string> args;
};

class SymUtils {
public:
  Logger logger_;
  myenv env_;
  explicit SymUtils(pid_t pid, Logger &logger, UprobeProfiler &profiler, myenv& env)
      : pid_(pid), logger_(logger), profiler_(profiler), env_(env){
    symbolizer_ = blaze_symbolizer_new();
  }

  std::vector<std::pair<std::string, size_t>>
  findSymbolOffsets(const std::string &symName, bool searchAllMappings = true,
                    bool exitOnFirstMatch = false);

  std::vector<StackFrame> getStackByAddrs(uint64_t *stack, size_t stack_sz);

  SymbolInfo getSymbolByAddr(size_t addr, bool parseArgs = false);

  ~SymUtils() {
    if (symbolizer_) {
      blaze_symbolizer_free(symbolizer_);
    }
  }

private:
  pid_t pid_;
  struct blaze_symbolizer *symbolizer_;
  std::map<size_t, SymbolInfo> cachedSyms_;
  
  UprobeProfiler& profiler_;

  bool findSymbolOffsetInFile(const std::string &elfPath,
                              const std::string &symbolName, size_t &symAddr);
  bool findSymbolOffsetInMMap(const pid_t pid, const MemoryMapping &mm,
                              const std::string &symName, size_t &addr);
};
} // namespace NeuTracer
