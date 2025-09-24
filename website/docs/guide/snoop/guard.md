# Guard: 自动资源管理工具类

## 概述

Guard 是一个 C++ 工具类，基于 RAII (Resource Acquisition Is Initialization) 设计模式，用于确保在代码块执行结束时自动执行清理操作。它能有效防止资源泄漏，并简化错误处理逻辑。

## 工作原理

Guard 类利用 C++ 对象生命周期管理机制，在构造时接收一个清理函数，并在析构时自动执行该函数，无论代码块是正常执行结束还是因异常退出。

## 核心特性

- 无需手动编写清理代码

- 即使发生异常，也能保证资源被正确释放

- 消除了多处返回点需要重复编写清理代码的问题

- 使资源管理更清晰，提高可维护性

- 禁用复制和移动操作，防止误用



## Guard 类实现

```cpp
#include <functional>
#include <utility>

class Guard {
public:
    explicit Guard(std::function<void()> cleanupFunction)
        : cleanupFunction_(std::move(cleanupFunction)) {}

    ~Guard() {
        if (cleanupFunction_) {
            cleanupFunction_();
        }
    }

    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    Guard(Guard&&) = delete;
    Guard& operator=(Guard&&) = delete;

private:
    std::function<void()> cleanupFunction_;
};
```

