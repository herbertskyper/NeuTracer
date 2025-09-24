// #include "vmlinux.h"
// #include <bpf/bpf_helpers.h>
// #include <bpf/bpf_tracing.h>
// #include <bpf/bpf_core_read.h>
// #define TASK_COMM_LEN 16

// char LICENSE[] SEC("license") = "GPL";

// enum mlx_event_type {
//     EVENT_MLX_ENTER = 0,
//     EVENT_MLX_EXIT = 1,
// };

// enum mlx_func_id {
//     MLX_REG_DM_MR = 0,
//     MLX_REG_USER_MR = 1,
//     MLX_REG_USER_MR_DMABUF = 2,
//     MLX_ALLOC_PD = 3,
//     MLX_ALLOC_MR = 4,
//     MLX_CREATE_CQ = 5,
//     MLX_CREATE_QP = 6,
//     MLX_CREATE_SRQ = 7,
// };

// struct mlx_event {
//     enum mlx_event_type type;
//     enum mlx_func_id func_id;
//     u32 pid;
//     u32 tgid;
//     char comm[TASK_COMM_LEN];
//     u64 timestamp;
    
//     // 函数参数和返回值
//     u64 arg1;
//     u64 arg2;
//     u64 arg3;
//     u64 ret_val;
// };

// struct {
//     __uint(type, BPF_MAP_TYPE_HASH);
//     __uint(max_entries, 256);
//     __type(key, u32);
//     __type(value, u32);
// } snoop_proc SEC(".maps");

// struct {
//     __uint(type, BPF_MAP_TYPE_RINGBUF);
//     __uint(max_entries, 256 * 1024);
// } events SEC(".maps");


// static struct mlx_event *create_mlx_event(enum mlx_event_type type, 
//                                            enum mlx_func_id func_id, 
//                                            u32 pid, u32 tgid) {
//     struct mlx_event *event = bpf_ringbuf_reserve(&events, sizeof(struct mlx_event), 0);
//     if (!event)
//         return NULL;
        
//     event->type = type;
//     event->func_id = func_id;
//     event->pid = pid;
//     event->tgid = tgid;
//     event->timestamp = bpf_ktime_get_ns();
//     bpf_get_current_comm(&event->comm, sizeof(event->comm));
    
//     return event;
// }

// // mlx5_ib_reg_dm_mr
// SEC("kprobe/mlx5_ib_reg_dm_mr")
// int BPF_KPROBE(mlx_reg_dm_mr_enter, struct ib_dm *dm, struct ib_mr_init_attr *attr) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_REG_DM_MR, pid, tgid);
//     if (!event)
//         return 0;

//     // 保存参数信息
//     event->arg1 = (u64)dm;
//     event->arg2 = (u64)attr;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_reg_dm_mr")
// int BPF_KRETPROBE(mlx_reg_dm_mr_exit, struct ib_mr *ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_REG_DM_MR, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_reg_user_mr
// SEC("kprobe/mlx5_ib_reg_user_mr")
// int BPF_KPROBE(mlx_reg_user_mr_enter, struct ib_pd *pd, struct ib_mr_init_attr *init_attr) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_REG_USER_MR, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)pd;
//     event->arg2 = (u64)init_attr;

//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_reg_user_mr")
// int BPF_KRETPROBE(mlx_reg_user_mr_exit, struct ib_mr *ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_REG_USER_MR, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_reg_user_mr_dmabuf
// SEC("kprobe/mlx5_ib_reg_user_mr_dmabuf")
// int BPF_KPROBE(mlx_reg_user_mr_dmabuf_enter, struct ib_pd *pd, struct ib_mr_init_attr *init_attr, struct dma_buf *dma_buf) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_REG_USER_MR_DMABUF, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)pd;
//     event->arg2 = (u64)init_attr;
//     event->arg3 = (u64)dma_buf;

//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_reg_user_mr_dmabuf")
// int BPF_KRETPROBE(mlx_reg_user_mr_dmabuf_exit, struct ib_mr *ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_REG_USER_MR_DMABUF, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_alloc_pd
// SEC("kprobe/mlx5_ib_alloc_pd")
// int BPF_KPROBE(mlx_alloc_pd_enter, struct ib_pd *pd) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_ALLOC_PD, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)pd;

//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_alloc_pd")
// int BPF_KRETPROBE(mlx_alloc_pd_exit, int ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_ALLOC_PD, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_alloc_mr
// SEC("kprobe/mlx5_ib_alloc_mr")
// int BPF_KPROBE(mlx_alloc_mr_enter, struct ib_pd *pd, enum ib_mr_type mr_type, u32 max_num_sg) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_ALLOC_MR, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)pd;
//     event->arg2 = (u64)mr_type;
//     event->arg3 = (u64)max_num_sg;

//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_alloc_mr")
// int BPF_KRETPROBE(mlx_alloc_mr_exit, struct ib_mr *ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_ALLOC_MR, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_create_cq
// SEC("kprobe/mlx5_ib_create_cq")
// int BPF_KPROBE(mlx_create_cq_enter, struct ib_cq *cq, const struct ib_cq_init_attr *attr) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_CREATE_CQ, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)cq;
//     event->arg2 = (u64)attr;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_create_cq")
// int BPF_KRETPROBE(mlx_create_cq_exit, int ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_CREATE_CQ, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_create_qp
// SEC("kprobe/mlx5_ib_create_qp")
// int BPF_KPROBE(mlx_create_qp_enter, struct ib_qp *qp, struct ib_qp_init_attr *attr) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_CREATE_QP, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)qp;
//     event->arg2 = (u64)attr;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_create_qp")
// int BPF_KRETPROBE(mlx_create_qp_exit, int ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_CREATE_QP, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// // mlx5_ib_create_srq
// SEC("kprobe/mlx5_ib_create_srq")
// int BPF_KPROBE(mlx_create_srq_enter, struct ib_srq *srq, struct ib_srq_init_attr *init_attr) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_ENTER, MLX_CREATE_SRQ, pid, tgid);
//     if (!event)
//         return 0;

//     event->arg1 = (u64)srq;
//     event->arg2 = (u64)init_attr;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }

// SEC("kretprobe/mlx5_ib_create_srq")
// int BPF_KRETPROBE(mlx_create_srq_exit, int ret) {
//     u64 pid_tgid = bpf_get_current_pid_tgid();
//     u32 pid = (u32)pid_tgid;
//     u32 tgid = (u32)(pid_tgid >> 32);

//     if (bpf_map_lookup_elem(&snoop_proc, &pid) == NULL &&
//         bpf_map_lookup_elem(&snoop_proc, &tgid) == NULL)
//         return 0;

//     struct mlx_event *event = create_mlx_event(EVENT_MLX_EXIT, MLX_CREATE_SRQ, pid, tgid);
//     if (!event)
//         return 0;

//     event->ret_val = (u64)ret;
    
//     bpf_ringbuf_submit(event, 0);
//     return 0;
// }