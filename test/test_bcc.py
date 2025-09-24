# #!/usr/bin/python3
# from bcc import BPF

# # 修改test_bcc.py
# bpf_text = """
# #include <uapi/linux/ptrace.h>

# // 尝试多个可能的PCIe操作函数
# int kprobe__pci_bus_read_config_byte(struct pt_regs *ctx) {
#     bpf_trace_printk("pci_read_config_byte被调用\\n");
#     return 0;
# }

# """

# # 加载BPF程序
# b = BPF(text=bpf_text)
# print("正在跟踪PCI配置空间读取... 按Ctrl+C退出")

# # 输出跟踪信息
# b.trace_print()
from bcc import BPF

bpf_text = """
int trace_nvlink_memcpy(struct pt_regs *ctx) {
    bpf_trace_printk("nvlink_memcpy called\\n");
    return 0;
}
"""

b = BPF(text=bpf_text)

# 关键：指定模块名为 "nvidia"
b.attach_kprobe(event="nvlink_memcpy", fn_name="trace_nvlink_memcpy", module="nvidia")

print("Tracing nvlink_memcpy in nvidia module... Ctrl-C to quit.")
b.trace_print()