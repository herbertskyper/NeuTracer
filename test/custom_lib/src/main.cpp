#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include <cstdlib>

#include "NeuTracer.h"


#include "cuda_operations.h"

int main() {
    NeuTracer::Tracer tracer(NeuTracer::UPROBE_CFG_PATH, "error");

        
    // 开始跟踪 - 确保配置支持CUDA内核和运行时API的追踪
    tracer.run();

    const int N = 1024;
    float a[N], b[N], c[N];
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    
    // 调用CUDA函数
    vectorAdd(a, b, c, N);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            success = false;
            break;
        }
    }
    
    if (success) {
        std::cout << "Vector addition succeeded!" << std::endl;
    } else {
        std::cout << "Vector addition failed!" << std::endl;
    }
    
    tracer.close();

    return 0;
}