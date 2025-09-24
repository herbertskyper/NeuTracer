// include/Logger.h
#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <unordered_map>
#include <fmt/core.h>

namespace NeuTracer {

class Logger {
public:
    enum LogLevel {
        INFO = 0,
        WARN = 1,
        ERROR = 2,
        NONE = 3,
    };

    std::string format_event_timestamp(uint64_t timestamp_ns) const;

    Logger(const std::string& level = "info");
    // void info(const std::string& message);
    // void debug(const std::string& message);
    // void error(const std::string& message);
        // 使用可变参数模板的新方法
    template<typename... Args>
    void info(const std::string& message, Args&&... args) {
        if (level <= INFO) {
            fmt::print("[{}] [+] ", get_timestamp());
            fmt::print(message, std::forward<Args>(args)...);
            fmt::print("\n");
            fflush(stdout);
        }
    }

    template<typename... Args>
    void warn(const std::string& message, Args&&... args) {
        if (level <= WARN) {
            fmt::print("[{}] \033[33m[W]\033[0m ", get_timestamp());
            fmt::print("\033[33m"); 
            fmt::print(message, std::forward<Args>(args)...);
            fmt::print("\033[0m\n");
            fflush(stdout);
        }
    }

    template<typename... Args>
    void error(const std::string& message, Args&&... args) {
        if (level <= ERROR) {
            fmt::print("[{}] \033[31m[E]\033[0m ", get_timestamp());
            fmt::print("\033[31m"); 
            fmt::print(message, std::forward<Args>(args)...);
            fmt::print("\033[0m\n");
            fflush(stdout);
        }
    }
    std::string format_duration(unsigned long);
    

private:
    LogLevel level;
    std::unordered_map<std::string, LogLevel> level_dict;
    std::string get_timestamp();
    int64_t mono_to_wall_delta_ns_;
};

} // namespace NeuTracer

#endif // LOGGER_H