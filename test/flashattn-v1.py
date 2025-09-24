import torch
import torch.nn as nn
import math
import time
import threading
import gc
from contextlib import contextmanager

# Flash Attention v1 使用不同的导入
try:
    from flash_attn.flash_attention import FlashAttention
    FLASH_ATTN_V1_AVAILABLE = True
except ImportError:
    try:
        # 备用导入路径
        from flash_attn import FlashAttention
        FLASH_ATTN_V1_AVAILABLE = True
    except ImportError:
        print("Flash Attention v1 not available, falling back to standard attention")
        FLASH_ATTN_V1_AVAILABLE = False

class FlashAttentionV1Layer(nn.Module):
    """使用 Flash Attention v1 的注意力层"""
    def __init__(self, embed_dim, num_heads, head_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else embed_dim // num_heads
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        # Flash Attention v1 初始化
        if FLASH_ATTN_V1_AVAILABLE:
            self.flash_attn = FlashAttention(
                softmax_scale=self.scale,
                attention_dropout=dropout
            )
        
        # 层归一化和前馈网络
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, causal=False, training=True):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, embed_dim]
            causal: 是否使用因果掩码（用于解码器）
            training: 是否为训练模式
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # 第一个残差连接 + 注意力
        x = self.layer_norm1(x)
        
        # 线性变换
        q = self.q_proj(x)
        k = self.k_proj(x)  
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 使用 Flash Attention v1 或标准注意力
        if FLASH_ATTN_V1_AVAILABLE:
            # Flash Attention v1 的正确调用方式
            # 需要将输入拼接成 qkv 格式: [batch_size, seq_len, 3, num_heads, head_dim]
            qkv = torch.stack([q, k, v], dim=2)  # [batch_size, seq_len, 3, num_heads, head_dim]
            
            # Flash Attention v1 forward 函数的正确参数
            # Flash Attention v1 返回一个元组 (output, softmax_lse)
            flash_output = self.flash_attn(
                qkv,  # 主要输入参数
                causal=causal
            )
            
            # 检查返回值类型并提取注意力输出
            if isinstance(flash_output, tuple):
                attn_output = flash_output[0]  # 第一个元素是注意力输出
            else:
                attn_output = flash_output
        else:
            # 标准注意力实现作为备选
            attn_output = self._standard_attention(q, k, v, causal=causal, training=training)
        
        # 合并多头输出
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        
        # 第一个残差连接
        x = residual + attn_output
        
        # 第二个残差连接 + FFN
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x
    
    def _standard_attention(self, q, k, v, causal=False, training=True):
        """标准注意力机制作为备选"""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力分数计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 因果掩码
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores.masked_fill_(mask.bool(), float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Dropout
        if training and self.dropout > 0:
            attn_weights = torch.dropout(attn_weights, p=self.dropout, train=training)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置回 [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output

class FlashAttentionV1Model(nn.Module):
    """使用 Flash Attention v1 的 Transformer 模型"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            FlashAttentionV1Layer(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, input_ids, causal=False, training=True):
        batch_size, seq_len = input_ids.shape
        
        # 创建位置索引
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 词嵌入和位置编码
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x) if training else x
        
        # 通过所有 Transformer 层
        for layer in self.layers:
            x = layer(x, causal=causal, training=training)
        
        # 最终层归一化和输出投影
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        
        return logits

@contextmanager
def memory_profiler():
    """内存使用监控上下文管理器"""
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    yield
    
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print(f"Memory - Start: {start_memory / 1024**3:.2f} GB")
    print(f"Memory - End: {end_memory / 1024**3:.2f} GB")
    print(f"Memory - Peak: {peak_memory / 1024**3:.2f} GB")
    print(f"Memory - Delta: {(end_memory - start_memory) / 1024**3:.2f} GB")

def debug_flash_attention():
    """调试 Flash Attention v1 的返回值"""
    if not FLASH_ATTN_V1_AVAILABLE:
        print("Flash Attention v1 not available for debugging")
        return
    
    print("Debugging Flash Attention v1 return values...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建小测试
    batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
    
    flash_attn = FlashAttention(softmax_scale=head_dim**-0.5, attention_dropout=0.0)
    flash_attn = flash_attn.to(device)
    
    # 创建测试输入
    qkv = torch.randn(batch_size, seq_len, 3, num_heads, head_dim, device=device, dtype=torch.float16)
    
    with torch.no_grad():
        output = flash_attn(qkv, causal=True)
        print(f"Flash Attention output type: {type(output)}")
        
        if isinstance(output, tuple):
            print(f"Tuple length: {len(output)}")
            for i, item in enumerate(output):
                print(f"  Item {i}: shape={getattr(item, 'shape', 'N/A')}, type={type(item)}")
        else:
            print(f"Single output shape: {output.shape}")

def long_inference_test_v1(duration_minutes=5):
    """Flash Attention v1 长时间推理测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Flash Attention v1 long inference test on {device} for {duration_minutes} minutes...")
    print(f"Flash Attention v1 Available: {FLASH_ATTN_V1_AVAILABLE}")
    
    # 先调试 Flash Attention 返回值
    debug_flash_attention()
    
    # 创建较大的模型
    model = FlashAttentionV1Model(
        vocab_size=32000,
        embed_dim=2048,
        num_heads=32,
        num_layers=24,
        max_seq_len=2048,
        dropout=0.1
    )
    model = model.half().to(device)
    model.eval()
    
    # 测试参数
    batch_size = 32
    seq_len = 2048
    end_time = time.time() + duration_minutes * 60
    
    iteration = 0
    total_tokens = 0
    start_time = time.time()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print("Starting inference loop...\n")
    
    with memory_profiler():
        with torch.no_grad():
            while time.time() < end_time:
                try:
                    # 创建随机输入
                    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                    
                    # 执行推理
                    torch.cuda.synchronize()
                    iter_start = time.time()
                    
                    logits = model(input_ids, causal=True, training=False)
                    
                    torch.cuda.synchronize()
                    iter_time = time.time() - iter_start
                    
                    iteration += 1
                    total_tokens += batch_size * seq_len
                    
                    # 每10次迭代打印一次统计信息
                    if iteration % 10 == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = total_tokens / elapsed
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        
                        print(f"Iter {iteration:4d} | Time: {iter_time*1000:6.1f}ms | "
                              f"Tokens/s: {tokens_per_sec:7.0f} | Memory: {memory_used:.2f}GB | "
                              f"Elapsed: {elapsed/60:.1f}min")
                    
                    # 偶尔清理内存
                    if iteration % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    print(f"Error in iteration {iteration}: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
    
    total_time = time.time() - start_time
    print(f"\nCompleted {iteration} iterations in {total_time/60:.1f} minutes")
    print(f"Average tokens per second: {total_tokens/total_time:.0f}")

def autoregressive_generation_test_v1():
    """Flash Attention v1 自回归生成测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting Flash Attention v1 autoregressive generation test on {device}...")
    
    # 创建模型
    model = FlashAttentionV1Model(
        vocab_size=32000,
        embed_dim=1024,
        num_heads=16,
        num_layers=12,
        max_seq_len=2048
    )
    model = model.half().to(device)
    model.eval()
    
    batch_size = 4
    prompt_len = 128
    max_new_tokens = 512
    
    # 创建初始prompt
    input_ids = torch.randint(0, 32000, (batch_size, prompt_len), device=device)
    
    print(f"Generating {max_new_tokens} tokens for {batch_size} sequences...")
    
    with torch.no_grad():
        generated_tokens = 0
        start_time = time.time()
        
        for step in range(max_new_tokens):
            try:
                # 获取当前序列长度
                current_len = input_ids.size(1)
                
                # 确保不超过最大长度
                if current_len >= model.max_seq_len:
                    print(f"Reached maximum sequence length {model.max_seq_len}")
                    break
                
                # 执行前向传播（使用因果掩码）
                logits = model(input_ids, causal=True, training=False)
                
                # 获取最后一个位置的logits
                next_token_logits = logits[:, -1, :]
                
                # 采样下一个token（使用top-k采样）
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = torch.softmax(top_k_logits / 0.8, dim=-1)  # 温度采样
                next_indices = torch.multinomial(probs, num_samples=1)
                next_tokens = torch.gather(top_k_indices, 1, next_indices)
                
                # 拼接新token
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                generated_tokens += batch_size
                
                # 每50步打印一次进度
                if (step + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = generated_tokens / elapsed
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    
                    print(f"Generated {step+1:3d}/{max_new_tokens} tokens | "
                          f"Speed: {tokens_per_sec:.0f} tokens/s | "
                          f"Memory: {memory_used:.2f}GB | "
                          f"Seq len: {current_len}")
                          
            except RuntimeError as e:
                print(f"Error in generation step {step}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
    
    total_time = time.time() - start_time
    final_tokens_per_sec = generated_tokens / total_time
    
    print(f"Generation completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average generation speed: {final_tokens_per_sec:.0f} tokens/s")
    print(f"Final sequence length: {input_ids.size(1)}")

def continuous_memory_pressure_test_v1(duration_minutes=3):
    """Flash Attention v1 持续内存压力测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting Flash Attention v1 continuous memory pressure test for {duration_minutes} minutes...")
    
    # 创建模型
    model = FlashAttentionV1Model(
        vocab_size=32000,
        embed_dim=1536,
        num_heads=24,
        num_layers=18,
        max_seq_len=1536
    )
    model = model.half().to(device)
    model.eval()
    
    end_time = time.time() + duration_minutes * 60
    iteration = 0
    
    # 变化的批次大小和序列长度来制造内存压力
    batch_configs = [
        (2, 1536), (4, 1024), (8, 768), (16, 512), (32, 256),
        (1, 2048), (6, 1200), (12, 600), (24, 300)
    ]
    
    print("Creating memory fragmentation through varying batch sizes...")
    
    with torch.no_grad():
        while time.time() < end_time:
            # 随机选择配置
            batch_size, seq_len = batch_configs[iteration % len(batch_configs)]
            
            try:
                # 创建输入
                input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                
                # 执行推理
                start_iter = time.time()
                logits = model(input_ids, causal=True, training=False)
                iter_time = time.time() - start_iter
                
                iteration += 1
                
                # 打印状态
                if iteration % 10 == 0:
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    
                    print(f"Iter {iteration:4d} | Batch: {batch_size:2d}x{seq_len:4d} | "
                          f"Time: {iter_time*1000:5.1f}ms | "
                          f"Mem: {memory_used:.2f}GB (cached: {memory_cached:.2f}GB)")
                
                # 偶尔强制垃圾收集
                if iteration % 25 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at iteration {iteration} with batch {batch_size}x{seq_len}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    print(f"Completed {iteration} iterations under memory pressure")

def compare_v1_vs_standard():
    """比较 Flash Attention v1 与标准注意力的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nComparing Flash Attention v1 vs Standard Attention on {device}...")
    
    # 测试配置
    configs = [
        {"seq_len": 512, "batch_size": 8},
        {"seq_len": 1024, "batch_size": 4},
        {"seq_len": 2048, "batch_size": 2},
    ]
    
    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        
        print(f"\nTesting sequence length {seq_len}, batch size {batch_size}")
        
        # 创建模型
        model = FlashAttentionV1Model(
            vocab_size=32000,
            embed_dim=1024,
            num_heads=16,
            num_layers=6,
            max_seq_len=seq_len
        )
        model = model.half().to(device)
        model.eval()
        
        # 创建输入
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            # 预热
            _ = model(input_ids, causal=True, training=False)
            
            # 测试
            iterations = 10
            start_time = time.time()
            
            for i in range(iterations):
                torch.cuda.synchronize()
                iter_start = time.time()
                
                logits = model(input_ids, causal=True, training=False)
                
                torch.cuda.synchronize()
                iter_time = time.time() - iter_start
                
                if i == 0:  # 第一次运行
                    print(f"  First iteration: {iter_time*1000:.1f}ms")
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            tokens_per_sec = (batch_size * seq_len * iterations) / total_time
            
            print(f"  Average time per iteration: {avg_time*1000:.1f}ms")
            print(f"  Tokens per second: {tokens_per_sec:.0f}")
            print(f"  Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
        # 清理
        del model
        gc.collect()
        torch.cuda.empty_cache()

def stress_test_different_configs_v1():
    """Flash Attention v1 不同配置的压力测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting Flash Attention v1 stress test with different configurations on {device}...")
    
    # 测试不同的配置
    configs = [
        {"name": "Small", "embed_dim": 512, "num_heads": 8, "num_layers": 6, "batch_size": 16, "seq_len": 512},
        {"name": "Medium", "embed_dim": 1024, "num_heads": 16, "num_layers": 12, "batch_size": 8, "seq_len": 1024},
        {"name": "Large", "embed_dim": 2048, "num_heads": 32, "num_layers": 24, "batch_size": 4, "seq_len": 2048},
        {"name": "Extra Large", "embed_dim": 4096, "num_heads": 64, "num_layers": 32, "batch_size": 2, "seq_len": 1024},
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing {config['name']} configuration:")
        print(f"  - Embed dim: {config['embed_dim']}")
        print(f"  - Heads: {config['num_heads']}")
        print(f"  - Layers: {config['num_layers']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Sequence length: {config['seq_len']}")
        print(f"  - Flash Attention v1 Available: {FLASH_ATTN_V1_AVAILABLE}")
        
        try:
            # 创建模型
            model = FlashAttentionV1Model(
                vocab_size=32000,
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                max_seq_len=config['seq_len']
            )
            model = model.half().to(device)
            model.eval()
            
            # 运行测试
            batch_size = config['batch_size']
            seq_len = config['seq_len']
            
            with memory_profiler():
                with torch.no_grad():
                    # 预热
                    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                    _ = model(input_ids, causal=True, training=False)
                    
                    # 实际测试
                    iterations = 20
                    start_time = time.time()
                    
                    for i in range(iterations):
                        try:
                            input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                            logits = model(input_ids, causal=True, training=False)
                            
                            if (i + 1) % 5 == 0:
                                elapsed = time.time() - start_time
                                tokens_processed = (i + 1) * batch_size * seq_len
                                tokens_per_sec = tokens_processed / elapsed
                                memory_used = torch.cuda.memory_allocated() / 1024**3
                                
                                print(f"  Iteration {i+1:2d}/{iterations} | "
                                      f"Tokens/s: {tokens_per_sec:.0f} | "
                                      f"Memory: {memory_used:.2f}GB")
                        except RuntimeError as e:
                            print(f"  Error in iteration {i+1}: {str(e)}")
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
                    
                    total_time = time.time() - start_time
                    total_tokens = iterations * batch_size * seq_len
                    avg_tokens_per_sec = total_tokens / total_time
                    
                    print(f"  Average speed: {avg_tokens_per_sec:.0f} tokens/s")
                    print(f"  Total time: {total_time:.2f}s")
                    print(f"  Memory peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
            
            # 清理内存
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import time
    time.sleep(5)
    print("Flash Attention v1 Performance Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU tests.")
        exit(1)
    
    # 1. 长时间推理测试 (1分钟)
    long_inference_test_v1(duration_minutes=2)
    
    # # 2. 自回归生成测试
    # autoregressive_generation_test_v1()

    # # 3. 不同配置压力测试
    # stress_test_different_configs_v1()
    
    print("\n" + "=" * 60)
    print("All Flash Attention v1 tests completed!")