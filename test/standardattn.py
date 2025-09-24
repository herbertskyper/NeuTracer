import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import math
from contextlib import contextmanager

class StandardAttentionLayer(nn.Module):
    """使用传统注意力机制的注意力层"""
    def __init__(self, embed_dim, num_heads, head_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim else embed_dim // num_heads
        self.dropout = dropout
        self.scale = (self.head_dim) ** -0.5
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        
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
        
        # 重塑为多头形式 [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 传统注意力计算
        attn_output = self._standard_attention(q, k, v, causal=causal, training=training)
        
        # 转置回 [batch_size, seq_len, num_heads, head_dim] 并合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
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
        """传统注意力机制实现"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 计算注意力分数 [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用因果掩码（如果需要）
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores.masked_fill_(mask.bool(), float('-inf'))
        
        # Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用 Dropout（如果在训练模式）
        if training:
            attn_weights = self.attn_dropout(attn_weights)
        
        # 应用注意力权重到值 [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output

class StandardAttentionModel(nn.Module):
    """使用传统注意力机制的 Transformer 模型"""
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
            StandardAttentionLayer(embed_dim, num_heads, dropout=dropout)
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        
        yield
        
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        print(f"Memory - Start: {start_memory / 1024**3:.2f} GB")
        print(f"Memory - End: {end_memory / 1024**3:.2f} GB")
        print(f"Memory - Peak: {peak_memory / 1024**3:.2f} GB")
        print(f"Memory - Delta: {(end_memory - start_memory) / 1024**3:.2f} GB")
    else:
        yield
        print("Memory profiling not available (no CUDA)")

def long_inference_test(duration_minutes=5):
    """长时间推理测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Standard Attention long inference test on {device} for {duration_minutes} minutes...")
    
    # 创建较大的模型
    model = StandardAttentionModel(
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
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    iter_start = time.time()
                    
                    logits = model(input_ids, causal=True, training=False)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    iter_time = time.time() - iter_start
                    
                    iteration += 1
                    total_tokens += batch_size * seq_len
                    
                    # 每10次迭代打印一次统计信息
                    if iteration % 10 == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = total_tokens / elapsed
                        
                        if torch.cuda.is_available():
                            memory_used = torch.cuda.memory_allocated() / 1024**3
                            print(f"Iter {iteration:4d} | Time: {iter_time*1000:6.1f}ms | "
                                  f"Tokens/s: {tokens_per_sec:7.0f} | Memory: {memory_used:.2f}GB | "
                                  f"Elapsed: {elapsed/60:.1f}min")
                        else:
                            print(f"Iter {iteration:4d} | Time: {iter_time*1000:6.1f}ms | "
                                  f"Tokens/s: {tokens_per_sec:7.0f} | "
                                  f"Elapsed: {elapsed/60:.1f}min")
                    
                    # 偶尔清理内存
                    if iteration % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except RuntimeError as e:
                    print(f"Error in iteration {iteration}: {e}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
    
    total_time = time.time() - start_time
    print(f"\nCompleted {iteration} iterations in {total_time/60:.1f} minutes")
    print(f"Average tokens per second: {total_tokens/total_time:.0f}")

def autoregressive_generation_test():
    """自回归生成测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting Standard Attention autoregressive generation test on {device}...")
    
    # 创建模型
    model = StandardAttentionModel(
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
                    
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        print(f"Generated {step+1:3d}/{max_new_tokens} tokens | "
                              f"Speed: {tokens_per_sec:.0f} tokens/s | "
                              f"Memory: {memory_used:.2f}GB | "
                              f"Seq len: {current_len}")
                    else:
                        print(f"Generated {step+1:3d}/{max_new_tokens} tokens | "
                              f"Speed: {tokens_per_sec:.0f} tokens/s | "
                              f"Seq len: {current_len}")
                              
            except RuntimeError as e:
                print(f"Error in generation step {step}: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    total_time = time.time() - start_time
    final_tokens_per_sec = generated_tokens / total_time
    
    print(f"Generation completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average generation speed: {final_tokens_per_sec:.0f} tokens/s")
    print(f"Final sequence length: {input_ids.size(1)}")

def stress_test_different_configs():
    """不同配置的压力测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting stress test with different configurations on {device}...")
    
    # 测试不同的配置（相比Flash Attention，减小一些配置以适应内存限制）
    configs = [
        {"name": "Small", "embed_dim": 512, "num_heads": 8, "num_layers": 6, "batch_size": 16, "seq_len": 512},
        {"name": "Medium", "embed_dim": 1024, "num_heads": 16, "num_layers": 12, "batch_size": 8, "seq_len": 1024},
        {"name": "Large", "embed_dim": 2048, "num_heads": 32, "num_layers": 24, "batch_size": 4, "seq_len": 2048},  # 减小seq_len
        {"name": "Extra Large", "embed_dim": 4096, "num_heads": 64, "num_layers": 32, "batch_size": 2, "seq_len": 1024},  # 进一步减小
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing {config['name']} configuration:")
        print(f"  - Embed dim: {config['embed_dim']}")
        print(f"  - Heads: {config['num_heads']}")
        print(f"  - Layers: {config['num_layers']}")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Sequence length: {config['seq_len']}")
        
        try:
            # 创建模型
            model = StandardAttentionModel(
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
                        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                        logits = model(input_ids, causal=True, training=False)
                        
                        if (i + 1) % 5 == 0:
                            elapsed = time.time() - start_time
                            tokens_processed = (i + 1) * batch_size * seq_len
                            tokens_per_sec = tokens_processed / elapsed
                            print(f"  Iteration {i+1:2d}/{iterations} | Tokens/s: {tokens_per_sec:.0f}")
                    
                    total_time = time.time() - start_time
                    total_tokens = iterations * batch_size * seq_len
                    avg_tokens_per_sec = total_tokens / total_time
                    
                    print(f"  Average speed: {avg_tokens_per_sec:.0f} tokens/s")
            
            # 清理内存
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def continuous_memory_pressure_test(duration_minutes=3):
    """持续内存压力测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStarting continuous memory pressure test for {duration_minutes} minutes...")
    
    # 创建模型（相比Flash Attention减小配置）
    model = StandardAttentionModel(
        vocab_size=32000,
        embed_dim=1024,  # 减小embed_dim
        num_heads=16,    # 减小num_heads
        num_layers=12,   # 减小num_layers
        max_seq_len=1024 # 减小max_seq_len
    )
    model = model.half().to(device)
    model.eval()
    
    end_time = time.time() + duration_minutes * 60
    iteration = 0
    
    # 变化的批次大小和序列长度来制造内存压力（减小配置）
    batch_configs = [
        (2, 1024), (4, 768), (8, 512), (16, 256), (32, 128),
        (1, 1024), (6, 512), (12, 256), (24, 128)
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
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_cached = torch.cuda.memory_reserved() / 1024**3
                        
                        print(f"Iter {iteration:4d} | Batch: {batch_size:2d}x{seq_len:4d} | "
                              f"Time: {iter_time*1000:5.1f}ms | "
                              f"Mem: {memory_used:.2f}GB (cached: {memory_cached:.2f}GB)")
                    else:
                        print(f"Iter {iteration:4d} | Batch: {batch_size:2d}x{seq_len:4d} | "
                              f"Time: {iter_time*1000:5.1f}ms")
                
                # 偶尔强制垃圾收集
                if iteration % 25 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at iteration {iteration} with batch {batch_size}x{seq_len}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    print(f"Completed {iteration} iterations under memory pressure")

def compare_attention_mechanisms():
    """比较不同序列长度下的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nComparing Standard Attention performance at different sequence lengths on {device}...")
    
    # 测试不同序列长度的性能特征
    seq_lengths = [128, 256, 512, 1024]
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    batch_size = 8
    
    for seq_len in seq_lengths:
        print(f"\n{'='*40}")
        print(f"Testing sequence length: {seq_len}")
        
        try:
            # 创建模型
            model = StandardAttentionModel(
                vocab_size=32000,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_seq_len=seq_len
            )
            model = model.half().to(device)
            model.eval()
            
            with torch.no_grad():
                # 预热
                input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                _ = model(input_ids, causal=True, training=False)
                
                # 性能测试
                iterations = 10
                total_time = 0
                
                for i in range(iterations):
                    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()
                    
                    logits = model(input_ids, causal=True, training=False)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    iter_time = time.time() - start_time
                    total_time += iter_time
                
                avg_time = total_time / iterations
                tokens_per_sec = (batch_size * seq_len) / avg_time
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    print(f"  Average time: {avg_time*1000:.1f}ms")
                    print(f"  Tokens/s: {tokens_per_sec:.0f}")
                    print(f"  Memory: {memory_used:.2f}GB")
                    print(f"  Memory per token: {memory_used*1024*1024/(batch_size*seq_len):.1f}KB")
                else:
                    print(f"  Average time: {avg_time*1000:.1f}ms")
                    print(f"  Tokens/s: {tokens_per_sec:.0f}")
                
                # 计算理论复杂度比较
                attention_ops = batch_size * num_heads * seq_len * seq_len
                print(f"  Attention operations: {attention_ops/1e6:.1f}M")
                print(f"  Ops per second: {attention_ops/avg_time/1e9:.1f}GOPS")
            
            # 清理
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Failed: {str(e)}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Standard Attention Performance Test")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 1. 长时间推理测试 (1分钟)
    long_inference_test(duration_minutes=2)
    
    # # 2. 自回归生成测试
    # autoregressive_generation_test()
    
    # # 3. 不同配置压力测试
    # stress_test_different_configs()
    
    print("\n" + "=" * 60)
    print("All Standard Attention tests completed!")
    print("\nNote: Standard attention has O(n²) memory complexity,")
    print("so configurations are scaled down compared to Flash Attention tests.")