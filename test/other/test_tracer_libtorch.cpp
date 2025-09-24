#include <torch/torch.h>
#include <iostream>
#include <unistd.h>
#include "NeuTracer.h"

// 定义一个简单的神经网络模型
struct SimpleNetImpl : torch::nn::Module {
    SimpleNetImpl() {
        // 创建一个具有1个输入特征和10个输出特征的全连接层
        fc1 = register_module("fc1", torch::nn::Linear(1, 8));
        fc2 = register_module("fc2", torch::nn::Linear(8, 4));
        fc3 = register_module("fc3", torch::nn::Linear(4, 1));
    }

    // 前向传播函数
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

TORCH_MODULE(SimpleNet);

int main() {
    NeuTracer::Tracer tracer(NeuTracer::UPROBE_CFG_PATH, "error");
        
    //tracer.clean();  // 清理之前的追踪数据
    tracer.run(); 

    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "Process ID: " << getpid() << std::endl;
    
    // 检查CUDA是否可用
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    // 设置设备（CPU或CUDA）
    torch::Device device = cuda_available ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (cuda_available ? "CUDA" : "CPU") << std::endl;
    
    // 创建一个简单的张量
    torch::Tensor tensor = torch::ones({2, 3});
    std::cout << "Created tensor: " << tensor << std::endl;
    
    // 操作张量
    torch::Tensor tensor2 = tensor * 2.0;
    std::cout << "Tensor multiplied by 2: " << tensor2 << std::endl;
    
    // 将张量移到指定设备
    tensor = tensor.to(device);
    std::cout << "Tensor moved to device: " << tensor.device() << std::endl;
    
    // 创建模型
    SimpleNet model;
    model->to(device);
    
    // 创建一些训练数据 (x, y) 数据点
    std::vector<float> x_values(100);
    std::vector<float> y_values(100);
    for (int i = 0; i < 100; i++) {
        x_values[i] = i * 0.1f;
        // 创建一个简单的函数关系: y = 2*x + 1 + 一些噪声
        y_values[i] = 2 * x_values[i] + 1 + ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    
    auto x_tensor = torch::from_blob(x_values.data(), {100, 1}).clone().to(torch::kFloat32).to(device);
    auto y_tensor = torch::from_blob(y_values.data(), {100, 1}).clone().to(torch::kFloat32).to(device);
    
    // 设置优化器
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));
    
    // 训练循环
    std::cout << "开始训练..." << std::endl;
    for (int epoch = 0; epoch < 100; epoch++) {
        // 前向传播
        torch::Tensor prediction = model->forward(x_tensor);
        
        // 计算损失
        torch::Tensor loss = torch::mse_loss(prediction, y_tensor);
        
        // 每10轮打印损失
        if (epoch % 10 == 0) {
            std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << std::endl;
        }
        
        // 反向传播和优化
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    
    // 测试模型
    model->eval();
    torch::NoGradGuard no_grad;
    
    // 创建一些测试数据
    std::vector<float> test_x = {0.5f, 1.0f, 1.5f, 5.0f, 10.0f};
    auto test_tensor = torch::from_blob(test_x.data(), {5, 1}).clone().to(torch::kFloat32).to(device);
    
    // 进行预测
    torch::Tensor test_pred = model->forward(test_tensor);
    std::cout << "测试值: [0.5, 1.0, 1.5, 5.0, 10.0]" << std::endl;
    std::cout << "预期值: [约2.0, 约3.0, 约4.0, 约11.0, 约21.0]" << std::endl;
    std::cout << "预测值: " << test_pred << std::endl;
    
    // 保存模型
    torch::save(model, "simple_net_model.pt");
    std::cout << "模型已保存到 simple_net_model.pt" << std::endl;
    
    // 尝试加载模型
    SimpleNet loaded_model;
    torch::load(loaded_model, "simple_net_model.pt");
    loaded_model->to(device);
    std::cout << "模型已从 simple_net_model.pt 加载" << std::endl;
    
    // 使用加载的模型进行预测
    torch::Tensor loaded_pred = loaded_model->forward(test_tensor);
    std::cout << "加载模型预测值: " << loaded_pred << std::endl;

    tracer.close();
    
    return 0;
}