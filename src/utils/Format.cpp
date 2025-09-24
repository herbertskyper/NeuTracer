#include <fstream>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

#include "utils/Format.h"
#include "client/tracer.h"

namespace NeuTracer {
    //  std::string format_event_timestamp(uint64_t timestamp_ns) {
    //     using namespace std::chrono;
    //     uint64_t now = timestamp_ns + uint64_t(Tracer::mono_to_wall_delta_ns_);
    //     system_clock::time_point tp{duration_cast<system_clock::duration>(nanoseconds(now))};
    //     auto event_time_t = system_clock::to_time_t(tp);
    //     auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;

    //     std::stringstream time_ss;
    //     time_ss << std::put_time(std::localtime(&event_time_t), "%Y-%m-%d %H:%M:%S")
    //             << '.' << std::setfill('0') << std::setw(3) << ms.count();
    //     return time_ss.str();
    // }
    std::string pid_to_comm(uint32_t pid) {
        std::string comm_file = "/proc/" + std::to_string(pid) + "/comm";
        std::ifstream comm(comm_file);
        if (!comm) return "";

        std::string name;
        std::getline(comm, name);
        return name;
    }

uint32_t run_command_get_pid(const std::string& command) {
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);
        exit(EXIT_FAILURE);
    }
    return pid;
}
// 将纳秒时间戳转换为可读格式（带毫秒微秒）
void format_timestamp(uint64_t ns, char *buffer, size_t buf_size) {
    time_t sec = ns / 1000000000;
    long nsec = ns % 1000000000;
    struct tm tm_info;
    
    localtime_r(&sec, &tm_info);
    strftime(buffer, buf_size, "%H:%M:%S", &tm_info);
    
    // 追加毫秒和微秒
    char msec_part[16];
    snprintf(msec_part, sizeof(msec_part), ".%03ld%03ld", 
             nsec / 1000000, (nsec % 1000000) / 1000);
    strncat(buffer, msec_part, buf_size - strlen(buffer) - 1);
}

// 将纳秒转换为人类可读时间长度
void format_duration(uint64_t ns, char *buffer, size_t buf_size) {
    if (ns < 1000) {
        snprintf(buffer, buf_size, "%" PRIu64 "ns", ns);
    } else if (ns < 1000000) {
        snprintf(buffer, buf_size, "%.3fµs", ns / 1000.0);
    } else if (ns < 1000000000) {
        snprintf(buffer, buf_size, "%.3fms", ns / 1000000.0);
    } else {
        snprintf(buffer, buf_size, "%.3fs", ns / 1000000000.0);
    }
}
}
    