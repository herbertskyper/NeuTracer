// src/Logger.cpp
#include "utils/Logger.h"
#include <iostream>
#include <chrono>
#include <sstream>  
#include <iomanip>  
#include <ctime>    


namespace NeuTracer {

Logger::Logger(const std::string& level) {
      auto wall_now = std::chrono::system_clock::now();
      auto mono_now = std::chrono::steady_clock::now();
      auto wall_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now.time_since_epoch()).count();
      auto mono_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(mono_now.time_since_epoch()).count();
      mono_to_wall_delta_ns_ = wall_epoch - mono_epoch;

    level_dict = {
        {"none", NONE},
        {"info", INFO},
        {"warn", WARN},
        {"error", ERROR}
    };
    
    auto it = level_dict.find(level);
    if (it != level_dict.end()) {
        this->level = it->second;
    } else {
        this->level = NONE;
    }
    fmt::print("Logger initialized with level: {}\n", level);
}

// 添加带时间戳的辅助函数
std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    // 使用 std::strftime 和 std::stringstream 代替 fmt::format
    char buffer[20];  // YYYY-MM-DD HH:MM:SS
    std::tm* timeinfo = std::localtime(&now_time_t);
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
    
    // 使用 stringstream 添加毫秒部分
    std::stringstream ss;
    ss << buffer << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

std::string Logger::format_duration(unsigned long duration_us) {
    if (duration_us < 1000) {
        // 小于1毫秒，显示为微秒
        return std::to_string(duration_us) + "µs";
    } else if (duration_us < 1000000) {
        // 小于1秒，显示为毫秒
        double ms = duration_us / 1000.0;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << ms << "ms";
        return ss.str();
    } else {
        // 大于1秒，显示为秒
        double s = duration_us / 1000000.0;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << s << "s";
        return ss.str();
    }
}

std::string Logger::format_event_timestamp(uint64_t timestamp_ns) const {
    using namespace std::chrono;
    uint64_t now = timestamp_ns + uint64_t(mono_to_wall_delta_ns_);
    system_clock::time_point tp(duration_cast<system_clock::duration>(nanoseconds(now)));
    auto event_time_t = system_clock::to_time_t(tp);
    auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;

    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&event_time_t), "%Y-%m-%d %H:%M:%S")
            << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return time_ss.str();
}

// 取消注释并实现这些函数
// void Logger::info(const std::string& message) {
//     if (level >= INFO) {
//         fmt::print("[{}] [+] {}\n", get_timestamp(), message);
//     }
// }

// void Logger::debug(const std::string& message) {
//     if (level >= DEBUG) {
//         fmt::print("[{}] [D] {}\n", get_timestamp(), message);
//     }
// }

// void Logger::error(const std::string& message) {
//     if (level >= ERROR) {
//         fmt::print("[{}] [E] {}\n", get_timestamp(), message);
//     }
// }



} // namespace NeuTracer