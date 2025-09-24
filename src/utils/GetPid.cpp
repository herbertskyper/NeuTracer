#include "utils/GetPid.h"
#include <fstream>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unordered_set>
#include <filesystem>
#include <sstream>
#include <vector>
namespace NeuTracer {

    std::vector<uint32_t> bfs_get_procs(uint32_t root_pid) {
        std::vector<uint32_t> pids;
        std::unordered_set<uint32_t> visited;  // 避免重复处理
        pids.push_back(root_pid);
        visited.insert(root_pid);
    
        for (size_t i = 0; i < pids.size(); i++) {
            std::string proc_dir = "/proc/" + std::to_string(pids[i]) + "/task";
            
            // 处理线程组（可选）
            if (!std::filesystem::exists(proc_dir)) {
                proc_dir = "/proc/" + std::to_string(pids[i]);
            }
    
            for (const auto& entry : std::filesystem::directory_iterator(proc_dir)) {
                if (!entry.is_directory()) continue;
                // C++11 does not have std::filesystem, so you need to use other methods for directory iteration.
                // If you want to keep this code C++11 compatible, consider using <dirent.h> for directory traversal.
                // The following line is just a comment and does not affect C++11 compatibility:
                //std::cout << "Checking dir: " << proc_dir << std::endl;
                std::string dirname = entry.path().filename();
                uint32_t current_pid;
                try {
                    current_pid = static_cast<uint32_t>(std::stoul(dirname));  // 使用 stoul 避免符号问题
                } catch (...) {
                    continue;  // 忽略无效 PID
                }
    
                if (visited.count(current_pid)) continue;
    
                std::string stat_file = entry.path() / "stat";
                std::ifstream stat(stat_file);
                if (!stat) continue;
    
                std::string line;
                if (!std::getline(stat, line)) continue;
    
                std::istringstream iss(line);
                std::vector<std::string> tokens;
                std::string token;
                while (iss >> token) {
                    tokens.push_back(token);
                }
    
                if (tokens.size() < 4) continue;  // 确保至少有 PPID
    
                try {
                    uint32_t ppid = static_cast<uint32_t>(std::stoul(tokens[3]));
                    if (ppid == pids[i]) {
                        pids.push_back(current_pid);
                        visited.insert(current_pid);
                    }
                } catch (...) {
                    continue;  // 忽略无效 PPID
                }
            }
        }
    
        return pids;
    }
    


}