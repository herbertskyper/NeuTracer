#include "NeuTracer.h"

#include <argp.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace NeuTracer;

myenv env;

static const struct argp_option opts[] = {
    {"verbose", 'v', nullptr, 0, "Verbose debug output", 0},
    {"pid", 'p', "PID", 0, "Trace process with given PID", 0},
    {"rb-count", 'r', "CNT", 0, "RingBuf max entries", 0},
    {"duration", 'd', "SEC", 0, "Trace for given number of seconds", 0},
    {"args", 'a', nullptr, 0, "Collect Kernel Launch Arguments", 0},
    {"stacks", 's', nullptr, 0, "Collect Kernel Launch Stacks", 0},
    {"grpc", 'g', nullptr, 0, "Enable gRPC server for real-time data streaming",
     0},
    {"server", 'S', "ADDR:PORT", 0,
     "gRPC server address (default: localhost:50051)", 0},
    {"trace", 't', "trace_modules", 0, "tracing modules (default: all)", 0},
    {}};

const char argp_program_doc[] =
    "NeuTrace.\n"
    "\n"
    "NeuTracer is a comprehensive eBPF-based performance monitoring and "
    "analysis system for AI/ML applications, providing real-time tracking of "
    "GPU operations, CPU utilization, memory allocation, and network I/O with "
    "intelligent anomaly detection.  \n"
    "\n"
    "USAGE:\n"
    "  ./neutracer -p PID [args]           # Trace existing process\n"
    "  ./neutracer profile COMMAND [args]  # Launch and trace program\n"
    "\n"
    "EXAMPLES:\n"
    "  ./neutracer -p 12345 -v -d 30\n"
    "  ./neutracer profile python train.py -a -s\n"
    "  ./neutracer profile ./my_app arg1 arg2 -g\n";

// 全局变量存储要启动的程序信息
static std::vector<std::string> target_command;
static bool profile_mode = false;

static error_t parse_arg(int key, char *arg, struct argp_state *state) {
  switch (key) {
  case 'v':
    env.verbose = true;
    break;
  case 'a':
    env.args = true;
    break;
  case 's':
    env.stacks = true;
    break;
  case 'p':
    errno = 0;
    env.pid = strtol(arg, nullptr, 10);
    if (errno || env.pid <= 0) {
      fmt::print(stderr, "Invalid pid: {}\n", arg);
      argp_usage(state);
    }
    break;
  case 'r':
    errno = 0;
    env.rb_count = strtol(arg, nullptr, 10);
    if (errno || env.rb_count == 0) {
      fmt::print(stderr, "Invalid ringbuf size: {}\n", arg);
      argp_usage(state);
    }
    break;
  case 'd':
    errno = 0;
    env.duration_sec = strtol(arg, nullptr, 10);
    if (errno) {
      fmt::print(stderr, "Invalid duration: {}\n", arg);
      argp_usage(state);
    }
    break;
  case 'g':
    env.grpc_enabled = true;
    break;
  case 'S':

    if (arg == nullptr || strlen(arg) == 0) {
      fmt::print(stderr, "gRPC server address cannot be empty\n");
      argp_usage(state);
    }

    // 验证地址格式是否为 addr:port
    {
      std::string addr_str(arg);
      size_t colon_pos = addr_str.find(':');

      if (colon_pos == std::string::npos) {
        fmt::print(
            stderr,
            "Invalid server address format: '{}'. Expected format: addr:port\n",
            arg);
        fmt::print(stderr, "Examples: localhost:50051, 192.168.1.100:8080\n");
        argp_usage(state);
      }

      std::string host = addr_str.substr(0, colon_pos);
      std::string port_str = addr_str.substr(colon_pos + 1);

      // 检查主机部分不为空
      if (host.empty()) {
        fmt::print(stderr, "Host part cannot be empty in address: '{}'\n", arg);
        argp_usage(state);
      }

      // 检查端口部分不为空且为有效数字
      if (port_str.empty()) {
        fmt::print(stderr, "Port part cannot be empty in address: '{}'\n", arg);
        argp_usage(state);
      }

      // 验证端口是否为有效数字
      char *endptr;
      long port = strtol(port_str.c_str(), &endptr, 10);
      if (*endptr != '\0' || port <= 0 || port > 65535) {
        fmt::print(
            stderr,
            "Invalid port number: '{}'. Port must be between 1 and 65535\n",
            port_str);
        argp_usage(state);
      }

      fmt::print("Using gRPC server: {}:{}\n", host, port);
    }

    env.server_addr = arg;
    break;
  case 't':
    if (arg == nullptr || strlen(arg) == 0) {
      fmt::print(stderr, "Tracing modules cannot be empty\n");
      argp_usage(state);
    }

    {
      std::string modules_str(arg);
      std::istringstream ss(modules_str);
      std::string module;

      env.trace_modules.gpu = false;
      env.trace_modules.cpu = false;
      env.trace_modules.kmem = false;
      env.trace_modules.net = false;
      env.trace_modules.io = false;
      env.trace_modules.func = false;
      env.trace_modules.python = false;
      env.trace_modules.syscall = false;
      env.trace_modules.nvlink = false;
      env.trace_modules.pcie = false;

      // 解析每个模块名
      while (std::getline(ss, module, ',')) {
        // 去除空格
        module.erase(0, module.find_first_not_of(" \t"));
        module.erase(module.find_last_not_of(" \t") + 1);

        // 转换为小写便于比较
        std::transform(module.begin(), module.end(), module.begin(), ::tolower);

        if (module == "gpu") {
          env.trace_modules.gpu = true;
          // fmt::print("Enabling GPU tracing\n");
        } else if (module == "cpu") {
          env.trace_modules.cpu = true;
          // fmt::print("Enabling CPU tracing\n");
        } else if (module == "memory" || module == "mem") {
          env.trace_modules.kmem = true;
          // fmt::print("Enabling Memory tracing\n");
        } else if (module == "network" || module == "net") {
          env.trace_modules.net = true;
          // fmt::print("Enabling Network tracing\n");
        } else if (module == "io") {
          env.trace_modules.io = true;
        } else if (module == "func" || module == "function") {
          // fmt::print("Enabling Function tracing\n");
          env.trace_modules.func = true;
        } else if (module == "python") {
          // fmt::print("Enabling Python tracing\n");
          env.trace_modules.python = true;
        } else if (module == "syscall") {
          // fmt::print("Enabling Syscall tracing\n");
          env.trace_modules.syscall = true;
        }
        else if (module == "nvlink") {
          env.trace_modules.nvlink = true;
        } else if (module == "pcie") {
          env.trace_modules.pcie = true;
        }

        else if (module == "all") {
          env.trace_modules.gpu = true;
          env.trace_modules.cpu = true;
          env.trace_modules.kmem = true;
          env.trace_modules.net = true;
          env.trace_modules.io = true;
          env.trace_modules.func = true;
          env.trace_modules.python = true;
          env.trace_modules.syscall = true;
          env.trace_modules.nvlink = true;
          env.trace_modules.pcie = true;
        } else if (!module.empty()) {
          fmt::print(stderr, "Unknown tracing module: {}\n", module);
          fmt::print(stderr, "Available modules: gpu, cpu, kmem/memory, "
                             "net/network, io, func/function, python, all\n");
          argp_usage(state);
        }
      }
    }
    break;

  case ARGP_KEY_ARG:
    // 处理非选项参数
    if (state->arg_num == 0) {
      // 第一个非选项参数
      if (strcmp(arg, "profile") == 0) {
        profile_mode = true;
        fmt::print("Profile mode enabled\n");
      } else {
        fmt::print(stderr,
                   "Unknown command: {}. Use 'profile' to launch and trace a "
                   "program\n",
                   arg);
        argp_usage(state);
      }
    } else if (profile_mode) {
      // profile 模式下的后续参数都是要执行的命令
      target_command.push_back(std::string(arg));
    } else {
      argp_usage(state);
    }
    break;
  case ARGP_KEY_END:
    // 参数解析结束时的验证
    if (profile_mode && target_command.empty()) {
      fmt::print(stderr, "Profile mode requires a command to execute\n");
      fmt::print(stderr, "Example: ./neutracer profile python script.py\n");
      argp_usage(state);
    }
    if (!profile_mode && env.pid == 0) {
      fmt::print(stderr, "Please specify PID with -p or use 'profile' mode\n");
      argp_usage(state);
    }
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

// 启动目标程序并返回其PID
pid_t launch_target_program() {
  if (target_command.empty()) {
    fmt::print(stderr, "No command specified for profile mode\n");
    return -1;
  }

  // 准备execvp参数
  std::vector<char *> argv_vec;
  for (const auto &arg : target_command) {
    argv_vec.push_back(const_cast<char *>(arg.c_str()));
  }
  argv_vec.push_back(nullptr);

  fmt::print("Launching target program: ");
  for (const auto &arg : target_command) {
    fmt::print("{} ", arg);
  }
  fmt::print("\n");

  pid_t child_pid = fork();
  if (child_pid == 0) {
    setsid();
    int null_fd = open("/dev/null", O_RDWR);
    if (null_fd >= 0) {
        // dup2(null_fd, STDIN_FILENO);
        dup2(null_fd, STDOUT_FILENO);
        dup2(null_fd, STDERR_FILENO);
        close(null_fd);
    }
    // 子进程：执行目标程序
    execvp(argv_vec[0], argv_vec.data());
    // 如果execvp返回，说明执行失败
    fmt::print(stderr, "Failed to execute {}: {}\n", target_command[0],
               strerror(errno));
    exit(1);
  } else if (child_pid > 0) {
    // 父进程：返回子进程PID
    fmt::print("Target program started with PID: {}\n", child_pid);
    std::this_thread::sleep_for(std::chrono::duration<double>(0.1));

    return child_pid;
  } else {
    // fork失败
    fmt::print(stderr, "Failed to fork: {}\n", strerror(errno));
    return -1;
  }
}

static const struct argp argp = {
    .options = opts,
    .parser = parse_arg,
    .doc = argp_program_doc,
};

int main(int argc, char *argv[]) {
  int err = argp_parse(&argp, argc, argv, 0, nullptr, nullptr);
  if (err) {
    fmt::print(stderr, "Failed to parse arguments\n");
    return err;
  }
  // 如果是profile模式，启动目标程序
  if (profile_mode) {
    pid_t target_pid = launch_target_program();
    if (target_pid <= 0) {
      fmt::print(stderr, "Failed to launch target program\n");
      return -1;
    }
    env.pid = target_pid;
    fmt::print("Tracing launched program with PID: {}\n", env.pid);
  }

  NeuTracer::Tracer tracer(NeuTracer::UPROBE_CFG_PATH, "info", env);
                           
  // 启动跟踪器
  tracer.run();

  if (env.duration_sec > 0) {
    sleep(env.duration_sec);
  } else {

    int status;

    if (profile_mode) {
      // 如果是profile模式，等待子进程正常结束
      waitpid(env.pid, &status, 0);
      fmt::print("Target program with PID {} has exited.\n", env.pid);
      if (WIFEXITED(status)) {
        fmt::print("Target program exited with code: {}\n",
                   WEXITSTATUS(status));
      } else if (WIFSIGNALED(status)) {
        fmt::print("Target program terminated by signal: {}\n",
                   WTERMSIG(status));
      }
    } else {
      while (kill(env.pid, 0) == 0) {
        sleep(1);
      }
    }
  }

  // 确保数据都处理完毕后再关闭
  sleep(1); // 给ring buffer一个处理剩余数据的机会

  // 最后清理资源
  tracer.close();

  fmt::print("Tracing completed.\n");
  return 0;
}