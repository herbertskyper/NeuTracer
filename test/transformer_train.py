import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import os
import json
import logging
from contextlib import contextmanager
import numpy as np
from typing import Optional, Dict, Any
import math

# 导入我们之前定义的注意力模块
import sys
sys.path.append('/home/skyper/neu-tracer/test')
from standardattn import StandardAttentionModel
try:
    from flashattn_v2 import FlashAttentionModel
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    print("Flash Attention not available, will only use standard attention")
    FLASH_ATTENTION_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticTextDataset(Dataset):
    """合成文本数据集，用于训练测试"""
    def __init__(self, vocab_size, seq_len, num_samples, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # 设置随机种子以确保可重现性
        torch.manual_seed(seed)
        
        # 生成合成数据
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 输入是除最后一个token外的所有token
        input_ids = self.data[idx, :-1]
        # 标签是除第一个token外的所有token（下一个token预测）
        labels = self.data[idx, 1:]
        return input_ids, labels

class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        # 模型配置
        self.vocab_size = 32000
        self.embed_dim = 1024
        self.num_heads = 16
        self.num_layers = 12
        self.max_seq_len = 1024
        self.dropout = 0.1
        
        # 训练配置
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 3
        self.warmup_steps = 1000
        self.max_grad_norm = 1.0
        
        # 数据配置
        self.train_samples = 10000
        self.val_samples = 1000
        
        # 其他配置
        self.save_dir = "/home/skyper/neu-tracer/tmp"
        self.log_interval = 100
        self.eval_interval = 500
        self.save_interval = 1000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True  # 使用混合精度训练
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

@contextmanager
def memory_profiler(name=""):
    """内存使用监控上下文管理器"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        
        yield
        
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        logger.info(f"{name} Memory Usage:")
        logger.info(f"  Start: {start_memory / 1024**3:.2f} GB")
        logger.info(f"  End: {end_memory / 1024**3:.2f} GB")
        logger.info(f"  Peak: {peak_memory / 1024**3:.2f} GB")
        logger.info(f"  Delta: {(end_memory - start_memory) / 1024**3:.2f} GB")
    else:
        yield

class LearningRateScheduler:
    """学习率调度器 - Warmup + Cosine Decay"""
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count < self.warmup_steps:
            # Warmup阶段：线性增长
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay阶段
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class TransformerTrainer:
    """Transformer 训练器"""
    def __init__(self, config: TrainingConfig, attention_type="standard"):
        self.config = config
        self.attention_type = attention_type
        self.device = torch.device(config.device)
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
        # 初始化数据
        self._init_data()
        
        # 初始化优化器和调度器
        self._init_optimizer()
        
        # 初始化混合精度训练
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Initialized trainer with {attention_type} attention")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
    
    def _init_model(self):
        """初始化模型"""
        if self.attention_type == "flash" and FLASH_ATTENTION_AVAILABLE:
            self.model = FlashAttentionModel(
                vocab_size=self.config.vocab_size,
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout
            )
        else:
            self.model = StandardAttentionModel(
                vocab_size=self.config.vocab_size,
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                max_seq_len=self.config.max_seq_len,
                dropout=self.config.dropout
            )
        
        self.model = self.model.to(self.device)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        
        self.model.apply(init_weights)
    
    def _init_data(self):
        """初始化数据集和数据加载器"""
        # 训练集
        train_dataset = SyntheticTextDataset(
            vocab_size=self.config.vocab_size,
            seq_len=self.config.max_seq_len,
            num_samples=self.config.train_samples,
            seed=42
        )
        
        # 验证集
        val_dataset = SyntheticTextDataset(
            vocab_size=self.config.vocab_size,
            seq_len=self.config.max_seq_len,
            num_samples=self.config.val_samples,
            seed=123
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        # 使用AdamW优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # 计算总步数
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        # 初始化学习率调度器
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            base_lr=self.config.learning_rate
        )
    
    def train_step(self, batch):
        """单步训练"""
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        
        self.model.train()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, causal=True, training=True)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(input_ids, causal=True, training=True)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # 优化器步进
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # 更新学习率
        current_lr = self.scheduler.step()
        
        return loss.item(), current_lr
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids, causal=True, training=False)
                        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                else:
                    logits = self.model(input_ids, causal=True, training=False)
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, step, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict(),
            'attention_type': self.attention_type
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_{self.attention_type}_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，也保存一份
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.config.save_dir, f'checkpoint_{self.attention_type}_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def train(self):
        """主训练循环"""
        logger.info("Starting training...")
        
        with memory_profiler("Training"):
            for epoch in range(self.config.num_epochs):
                epoch_start_time = time.time()
                epoch_loss = 0
                num_batches = 0
                
                for batch_idx, batch in enumerate(self.train_loader):
                    batch_start_time = time.time()
                    
                    # 训练步骤
                    loss, lr = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    self.global_step += 1
                    
                    batch_time = time.time() - batch_start_time
                    tokens_per_sec = (self.config.batch_size * (self.config.max_seq_len - 1)) / batch_time
                    
                    # 记录日志
                    if self.global_step % self.config.log_interval == 0:
                        memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        logger.info(
                            f"Epoch {epoch+1}/{self.config.num_epochs} | "
                            f"Step {self.global_step} | "
                            f"Loss: {loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tokens/s: {tokens_per_sec:.0f} | "
                            f"Memory: {memory_used:.2f}GB"
                        )
                    
                    # 验证
                    if self.global_step % self.config.eval_interval == 0:
                        val_loss = self.validate()
                        logger.info(f"Validation loss: {val_loss:.4f}")
                        
                        # 保存检查点
                        if self.global_step % self.config.save_interval == 0:
                            self.save_checkpoint(epoch, self.global_step, val_loss)
                
                # Epoch结束统计
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / num_batches
                
                logger.info(
                    f"Epoch {epoch+1} completed | "
                    f"Time: {epoch_time:.1f}s | "
                    f"Avg Loss: {avg_epoch_loss:.4f}"
                )
                
                # Epoch结束验证
                val_loss = self.validate()
                logger.info(f"End of epoch validation loss: {val_loss:.4f}")
                
                # 保存epoch检查点
                self.save_checkpoint(epoch, self.global_step, val_loss)
        
        logger.info("Training completed!")

def compare_attention_training():
    """比较不同注意力机制的训练性能"""
    config = TrainingConfig()
    
    # 调整配置以适应比较测试
    config.num_epochs = 1
    config.train_samples = 2000
    config.val_samples = 200
    config.log_interval = 50
    config.eval_interval = 200
    config.save_interval = 500
    
    results = {}
    
    # 测试标准注意力
    logger.info("=" * 60)
    logger.info("Training with Standard Attention")
    logger.info("=" * 60)
    
    std_trainer = TransformerTrainer(config, attention_type="standard")
    start_time = time.time()
    std_trainer.train()
    std_time = time.time() - start_time
    
    results['standard'] = {
        'training_time': std_time,
        'best_val_loss': std_trainer.best_val_loss,
        'memory_peak': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }
    
    # 清理内存
    del std_trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试Flash Attention（如果可用）
    if FLASH_ATTENTION_AVAILABLE:
        logger.info("=" * 60)
        logger.info("Training with Flash Attention")
        logger.info("=" * 60)
        
        flash_trainer = TransformerTrainer(config, attention_type="flash")
        start_time = time.time()
        flash_trainer.train()
        flash_time = time.time() - start_time
        
        results['flash'] = {
            'training_time': flash_time,
            'best_val_loss': flash_trainer.best_val_loss,
            'memory_peak': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
        
        del flash_trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 打印比较结果
    logger.info("=" * 60)
    logger.info("Training Comparison Results")
    logger.info("=" * 60)
    
    for attention_type, metrics in results.items():
        logger.info(f"{attention_type.upper()} Attention:")
        logger.info(f"  Training time: {metrics['training_time']:.1f}s")
        logger.info(f"  Best val loss: {metrics['best_val_loss']:.4f}")
        logger.info(f"  Peak memory: {metrics['memory_peak']:.2f}GB")
    
    if len(results) > 1:
        std_time = results['standard']['training_time']
        flash_time = results['flash']['training_time']
        speedup = std_time / flash_time
        
        std_memory = results['standard']['memory_peak']
        flash_memory = results['flash']['memory_peak']
        memory_ratio = std_memory / flash_memory if flash_memory > 0 else float('inf')
        
        logger.info(f"\nFlash Attention vs Standard:")
        logger.info(f"  Speed improvement: {speedup:.2f}x")
        logger.info(f"  Memory reduction: {memory_ratio:.2f}x")
    
    return results

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建配置
    config = TrainingConfig()
    
    # 保存配置
    with open(os.path.join(config.save_dir, 'training_config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # 选择训练类型
    import argparse
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument('--attention', choices=['standard', 'flash', 'compare'], 
                       default='standard', help='Attention mechanism to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    # 更新配置
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    
    if args.attention == 'compare':
        # 比较不同注意力机制
        compare_attention_training()
    else:
        # 单独训练
        trainer = TransformerTrainer(config, attention_type=args.attention)
        trainer.train()

if __name__ == "__main__":
    import os
    import time
    current_pid = os.getpid()
    print(f"当前进程 PID: {current_pid}")
    time.sleep(5)
    main()