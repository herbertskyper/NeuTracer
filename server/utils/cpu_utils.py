import subprocess
import psutil
import queue
import time

def pid_to_comm(pid):
    """根据pid查找/proc/pid/comm获取comm

    Args:
        pid (int): PID号

    Returns:
        string: comm
    """
    try:
        comm = open("/proc/%d/comm" % pid, "r").read()
        return comm.replace("\n", "")
    except IOError:
        return str(pid)

def bfs_get_procs(snoop_pid):
    """使用广度优先搜索获取snoop_pid进程下的所有子进程、孙子进程...

    Args:
        snoop_pid (int): 根节点进程pid

    Returns:
        list: 进程树下所有进程的pid
    """
    proc = psutil.Process(snoop_pid)
    proc_queue = queue.Queue()
    proc_queue.put(proc)
    pid_list = []
    while not proc_queue.empty():
        proc = proc_queue.get()
        pid_list.append(proc.pid)
        list(map(proc_queue.put, proc.children()))
    
    return pid_list