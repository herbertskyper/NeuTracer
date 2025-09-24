# Logger - 简洁灵活的日志记录工具

## 概述

Logger 是 NEU-Trace 项目中的日志记录工具类。

## 主要特性

- 多级别日志支持：INFO、WARN、ERROR 和 NONE 四个日志级别

- 彩色输出：不同级别使用不同颜色，提高可读性

- 时间戳：每条日志自动附加精确时间戳（精确到毫秒）

- 格式化输出：支持类似 printf 风格的格式化字符串

- 可变参数模板：灵活支持各种类型的参数传递

- 持久化：自动刷新输出缓冲区，确保日志及时写入

- 时间格式化：支持微秒、毫秒和秒的智能单位转换

## 使用方法

### 初始化

```cpp
Logger logger("log.txt");  // 将日志输出到文件 log.txt
```

### 记录不同级别的日志

```cpp
logger.info("This is an info message: {}", some_variable);
logger.warn("This is a warning message: {}", some_variable);
logger.error("This is an error message: {}", some_variable);
```

### 格式化时间间隔

```cpp
logger.info("Elapsed time: {:.2f} ms", elapsed_time);
```

### 日志级别控制

日志级别按严重程度递增：

- INFO (0): 一般信息，调试和正常操作

- WARN (1): 警告信息，可能存在问题但不影响程序继续运行

- ERROR (2): 错误信息，表示发生了严重问题

- NONE (3): 关闭所有日志输出

每个级别仅显示大于或等于当前设置级别的日志。例如，设置 WARN 级别时，将显示 WARN 和 ERROR 日志，但不显示 INFO 日志。

## 示例代码

```cpp
#include "logger.h"

int main() {
    Logger logger("log.txt");  // 初始化日志记录器
    logger.set_level(Logger::INFO);  // 设置日志级别为 INFO

    int some_variable = 42;
    logger.info("This is an info message: {}", some_variable);
    logger.warn("This is a warning message: {}", some_variable);
    logger.error("This is an error message: {}", some_variable);

    double elapsed_time = 123.456;
    logger.info("Elapsed time: {:.2f} ms", elapsed_time);

    return 0;
}


```
