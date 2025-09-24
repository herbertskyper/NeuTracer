# NeuTracer 服务器

NeuTracer 服务器是一个高性能的跟踪数据收集、处理和分析系统，专为处理来自 NeuTracer 客户端的各种系统性能数据而设计。服务器采用 gRPC 通信协议，支持实时数据处理和 Prometheus 指标导出，提供了全方位的系统性能监控解决方案。

## 功能特点

- **多维度性能数据收集**：支持 CPU、内存、I/O、网络、函数调用和 GPU 活动的跟踪数据
- **实时数据处理**：高性能的事件处理管道，支持批量数据处理
- **异常检测**：识别各类性能异常和资源使用问题
- **Prometheus 集成**：内置 Prometheus 指标导出，便于实时监控和告警
- **可视化数据导出**：生成兼容 Perfetto/Chrome Tracing 的可视化文件
- **CSV 数据导出**：提供结构化数据导出，便于进一步分析
- **自动化资源管理**：支持非活动超时和智能文件管理
- **可扩展架构**：模块化设计，易于添加新的数据处理器和分析功能

## 快速开始

### 前提条件

- Python 3.8+
- pip 包管理器
- gRPC 和 Protocol Buffers

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务器

```bash
python main.py --port 50051 --metrics-port 9091 --timeout 3600
```

参数说明：

- `--port`: gRPC 服务监听端口（默认：50051）
- `--metrics-port`: Prometheus 指标服务端口（默认：9091）
- `--timeout`: 非活动自动关闭超时时间（默认：60 秒）

也可以通过环境变量配置：

```bash
export SERVICE_PORT=50051
export METRICS_PORT=9091
export INACTIVITY_TIMEOUT=3600
python main.py
```

## 系统架构

### 核心组件

1. **TracerServicer**：gRPC 服务实现，处理客户端请求
2. **基础处理器 (BaseTraceProcessor)**：抽象基类，定义通用处理逻辑
3. **专用处理器**：
   - FunctionTraceProcessor：函数调用跟踪处理
   - CPUTraceProcessor：CPU 使用情况处理
   - MemTraceProcessor：内存分配和使用处理
   - NetworkTraceProcessor：网络活动处理
   - IOTraceProcessor：I/O 操作处理
   - GPUTraceProcessor：GPU 活动处理
4. **文件管理器**：管理输出文件的创建、写入和关闭
5. **活动监控器**：跟踪服务活动状态，处理超时关闭

### 数据流

```
客户端 → gRPC 请求 → TracerServicer → 专用处理器 → 文件输出/Prometheus指标
```

## Prometheus 指标

服务器导出多种 Prometheus 指标，便于实时监控和告警：

- 函数调用次数与执行时间
- CPU 利用率与核心使用分布
- 内存分配、释放和使用情况
- 网络流量、连接和协议分布
- I/O 操作数量、延迟和设备使用
- 异常检测结果和严重程度
- 服务器状态和性能指标


## 输出文件

服务器生成多种输出文件，存储在 output 目录中：

- `*.trace.json`：Perfetto/Chrome Tracing 兼容的可视化数据
- `*.csv`：按资源类型分类的结构化数据

## 开发扩展

### 添加新的处理器

1. 创建继承自 `BaseTraceProcessor` 的新类
2. 实现 `process_traces()` 方法
3. 在 `TracerServicer` 中初始化和使用新处理器

### 添加新的 Prometheus 指标

在 `prometheus_metrics.py` 中定义新的指标，并在相应的处理器中更新这些指标。

## 安全性考虑

默认配置下，服务器使用不安全的 gRPC 连接。在生产环境中，建议：

1. 配置 TLS/SSL 证书以启用安全连接
2. 实现适当的认证机制
3. 限制服务访问权限和网络暴露

