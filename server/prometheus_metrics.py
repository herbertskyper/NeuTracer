from prometheus_client import Counter, Gauge, Histogram


TRACE_COUNTER = Counter('neutracer_traces_total', 'Total number of traces received', ['type'])
BATCH_SIZE = Histogram('neutracer_batch_size', 'Size of trace batches', ['type'])

ACTIVE_CLIENTS = Gauge('neutracer_active_clients', 'Number of active clients')
PROCESSING_TIME = Histogram('neutracer_processing_time_seconds', 'Time spent processing trace batch')
INACTIVITY_TIME = Gauge('neutracer_inactivity_time_seconds', 'Time since last activity')

# GPU指标定义
GPU_MEM_SIZE = Gauge('neutracer_gpu_memory_size_bytes',
                    'Size of GPU memory operation in bytes',
                    ['pid'])
GPU_MEM_TRANS_RATE = Gauge('neutracer_gpu_memory_transfer_rate_Mbytes_per_second','GPU memory transfer rate in Mbytes per second',
                         ['pid', 'kind_str'])
GPU_KERNEL_FUNCTION_COUNT = Gauge('neutracer_gpu_kernel_function_count',
                                 'Count of GPU kernel functions executed',
                                 ['pid', 'kernel_name'])



# 函数调用相关指标
FUNCTION_CALLS = Gauge('neutracer_function_calls_total', 'Total number of function calls', ['function', 'process'])
FUNCTION_DURATION = Histogram('neutracer_function_duration_seconds', 'Function execution duration in seconds',
                             ['function', 'process'], buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
# SLOW_FUNCTION_CALLS = Counter('neutracer_slow_function_calls_total', 'Number of slow function calls', 
#                              ['function', 'process', 'threshold'])

# 函数调用状态摘要
ACTIVE_FUNCTIONS = Gauge('neutracer_active_functions', 'Number of currently executing functions', ['process'])
FUNCTION_STATS_SUMMARY = Gauge('neutracer_function_stats', 'Function statistics', 
                               ['function', 'process', 'metric'])

# CPU 基本指标
CPU_UTILIZATION = Gauge('neutracer_cpu_utilization_percent', 
                        'CPU utilization percentage', 
                        ['pid', 'process'])
CPU_ONCPU_TIME = Gauge('neutracer_cpu_on_time_microseconds', 
                         'Total time spent on CPU in microseconds', 
                         ['pid', 'process'])
CPU_OFFCPU_TIME = Gauge('neutracer_cpu_off_time_microseconds', 
                          'Total time spent off CPU in microseconds', 
                          ['pid', 'process'])

# CPU 调度指标
CPU_MIGRATION_COUNT = Gauge('neutracer_cpu_migrations_total', 
                             'Total number of CPU migrations', 
                             ['pid', 'process'])
CPU_NUMA_MIGRATION_COUNT = Gauge('neutracer_numa_migrations_total', 
                                  'Total number of cross-NUMA migrations', 
                                  ['pid', 'process'])
# CPU_CONTEXT_SWITCH_RATE = Gauge('neutracer_context_switches_per_second', 
#                                'Rate of context switches per second', 
#                                ['pid', 'process'])

# # CPU 使用分布
# CPU_HOTSPOT_PERCENTAGE = Gauge('neutracer_cpu_hotspot_percentage', 
#                               'Percentage of time spent on hotspot CPU', 
#                               ['pid', 'process'])



# IO 基本指标
IO_THROUGHPUT = Gauge('neutracer_io_throughput_bytes', 
                     'IO throughput in bytes per second', 
                     ['pid', 'operation'])
IO_LATENCY = Gauge('neutracer_io_latency_ms', 
                  'IO operation latency in milliseconds', 
                  ['pid', 'operation'])
IO_OPS_RATE = Gauge('neutracer_io_ops_per_second', 
                   'IO operations per second', 
                   ['pid', 'operation'])

IO_TOTAL_BYTES = Gauge('neutracer_io_total_bytes', 
                      'Total bytes transferred', 
                      ['pid', 'operation'])




# Memory基本指标
MEM_USAGE = Gauge('neutracer_memory_usage_bytes', 
                 'Current memory usage in bytes', 
                 ['pid'])
MEM_PEAK = Gauge('neutracer_memory_peak_bytes', 
                'Peak memory usage in bytes', 
                ['pid'])
MEM_ALLOC_RATE = Gauge('neutracer_memory_allocations_per_second', 
                      'Memory allocations per second', 
                      ['pid'])
MEM_FREE_RATE = Gauge('neutracer_memory_frees_per_second', 
                     'Memory deallocations per second', 
                     ['pid'])
MEM_TOTAL_ALLOCS = Gauge('neutracer_memory_total_allocations', 
                        'Total memory allocations', 
                        ['pid'])
MEM_TOTAL_FREES = Gauge('neutracer_memory_total_frees', 
                       'Total memory deallocations', 
                       ['pid'])

MEM_ALLOCATION_SIZE = Histogram('neutracer_memory_allocation_size_bytes',
                              'Distribution of memory allocation sizes',
                              ['pid'],
                              buckets=[128, 512, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304])


# Network基本指标
NET_TX_BYTES_RATE = Gauge('neutracer_network_tx_bytes_rate', 
                          'Network transmit rate in bytes per second', 
                          ['pid', 'process'])
NET_RX_BYTES_RATE = Gauge('neutracer_network_rx_bytes_rate', 
                          'Network receive rate in bytes per second', 
                          ['pid', 'process'])
NET_TX_PACKETS_RATE = Gauge('neutracer_network_tx_packets_rate', 
                           'Network transmit packet rate', 
                           ['pid', 'process'])
NET_RX_PACKETS_RATE = Gauge('neutracer_network_rx_packets_rate', 
                           'Network receive packet rate', 
                           ['pid', 'process'])

NET_TOTAL_TX_BYTES = Gauge('neutracer_network_total_tx_bytes', 
                           'Total transmitted bytes', 
                           ['pid', 'process'])
NET_TOTAL_RX_BYTES = Gauge('neutracer_network_total_rx_bytes', 
                           'Total received bytes', 
                           ['pid', 'process'])
NET_TOTAL_TX_PACKETS = Gauge('neutracer_network_total_tx_packets', 
                            'Total transmitted packets', 
                            ['pid', 'process'])
NET_TOTAL_RX_PACKETS = Gauge('neutracer_network_total_rx_packets', 
                            'Total received packets', 
                            ['pid', 'process'])

NET_ACTIVE_CONNECTIONS = Gauge('neutracer_network_active_connections', 
                              'Number of active network connections', 
                              ['pid', 'process'])
NET_LISTENING_PORTS = Gauge('neutracer_network_listening_ports', 
                           'Number of listening ports', 
                           ['pid', 'process'])
