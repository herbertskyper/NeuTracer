// #include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h> // 添加对fork()的支持
#include <sys/wait.h> // 添加对waitpid()的支持
#include <cstdlib> // 对exit()的支持

#include "NeuTracer.h"

// 使用nlohmann/json库的命名空间
using json = nlohmann::json;

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

// 子进程函数，用于在单独进程中执行add操作
void process_add_function() {
    while (true) {
        int a = 10;
        int b = 20;
        int result = add(a, b);
        std::cout << "Child Process (PID: " << getpid() << "): add result = " << result << std::endl;
        sleep(1); // 暂停1秒
    }
    exit(0); // 正常退出
}

int main() {
    // 创建Tracer实例
    NeuTracer::Tracer tracer(NeuTracer::UPROBE_CFG_PATH, "error");
                           
    // 开始跟踪
    tracer.run();
    
    // 创建子进程
    pid_t pid = fork();
    
    if (pid < 0) {
        // fork失败
        std::cerr << "Fork failed!" << std::endl;
        return 1;
    } else if (pid == 0) {
        // 子进程代码
        std::cout << "Child process started with PID: " << getpid() << std::endl;
        process_add_function();
        // 子进程不会执行到这里，因为process_add_function中有无限循环
    } else {
        // 父进程代码
        std::cout << "Parent process, child PID: " << pid << std::endl;
        
        // 主进程继续执行原来的循环
        while (true) {
            // 模拟跟踪数据
            int a = 1;
            int b = 2;
            int c = add(a, b);
            std::cout << "Main Process (PID: " << getpid() << "): c=" << c << std::endl;
            c = sub(a, b);
            std::cout << "Main Process (PID: " << getpid() << "): c=" << c << std::endl;
            sleep(1); // 暂停1秒
        }

        // 理论上这段代码不会被执行到，因为上面是无限循环
        int status;
        waitpid(pid, &status, 0); // 等待子进程结束
        
        // 停止跟踪
        tracer.close();
    }

    return 0;
}