#ifndef FORMAT_H
#define FORMAT_H

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <time.h>
#include <sys/time.h>
#include <inttypes.h>
#include <unistd.h>


namespace NeuTracer
{
    /**
     * @brief 将事件时间戳转换为格式化的时间字符串
     * @param timestamp_ns 纳秒级时间戳
     * @return 格式化的时间字符串 "YYYY-MM-DD HH:MM:SS.mmm"
     */
    std::string format_event_timestamp(uint64_t timestamp_ns);
    std::string pid_to_comm(uint32_t pid);
    uint32_t run_command_get_pid(const std::string& command);
    void format_timestamp(uint64_t ns, char *buffer, size_t buf_size);
    void format_duration(uint64_t ns, char *buffer, size_t buf_size) ;
} // namespace name

#endif // FORMAT_H