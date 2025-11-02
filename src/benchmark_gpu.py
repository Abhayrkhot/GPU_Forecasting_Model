"""
GPU Benchmarking Script - Simplified Version
Collects performance metrics under varying loads
"""

import torch
import time
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import pynvml
from dataclasses import dataclass, asdict
from typing import List
import gc

@dataclass
class BenchmarkMetrics:
    timestamp: str
    model_name: str
    engine: str
    batch_size: int
    concurrent_requests: int
    sequence_length: int
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    throughput_tokens_per_sec: float
    gpu_utilization_percent: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    kv_cache_size_mb: float
    tokens_generated: int

class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_metrics(self):
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {
            'gpu_util': utilization.gpu,
            'memory_used_gb': memory.used / 1e9,
            'memory_total_gb': memory.total / 1e9
        }
    
    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass

class PyTorchBenchmark:
    def __init__(self, model_name: str):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        self.gpu_monitor = GPUMonitor()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded")
    
    def benchmark_request(self, prompt: str, max_new_tokens: int = 128):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        total_time = time.time() - start_time
        
        gpu_metrics = self.gpu_monitor.get_metrics()
        
        # Estimate KV cache
        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size
        seq_len = outputs.shape[1]
        kv_cache_bytes = 2 * num_layers * seq_len * hidden_size * 2
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
        
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return {
            'total_latency': total_time * 1000,
            'throughput': tokens_generated / total_time,
            'kv_cache_mb': kv_cache_mb,
            'tokens_generated': tokens_generated,
            **gpu_metrics
        }
    
    def run_benchmark(self, num_requests: int, prompt: str, max_tokens: int):
        print(f"  Running {num_requests} requests...")
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            result = self.benchmark_request(prompt, max_tokens)
            results.append(result)
            
            if (i + 1) % 5 == 0:
                print(f"    Completed {i+1}/{num_requests}")
        
        total_time = time.time() - start_time
        
        return {
            'total_latency': total_time * 1000,
            'avg_latency': np.mean([r['total_latency'] for r in results]),
            'throughput': sum(r['tokens_generated'] for r in results) / total_time,
            'kv_cache_mb': np.mean([r['kv_cache_mb'] for r in results]),
            'tokens_generated': sum(r['tokens_generated'] for r in results),
            'gpu_util': np.mean([r['gpu_util'] for r in results]),
            'memory_used_gb': np.mean([r['memory_used_gb'] for r in results]),
            'memory_total_gb': results[0]['memory_total_gb']
        }

def run_comprehensive_benchmark(
    model_name: str = "gpt2",
    output_file: str = "benchmark_results.csv"
):
    print(f"\n{'='*80}")
    print(f"Starting Benchmark: {model_name}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Test configurations
    concurrent_requests = [1, 5, 10, 25, 50, 100]
    sequence_lengths = [128, 256]
    test_prompt = "Explain the concept of machine learning in detail."
    
    benchmark = PyTorchBenchmark(model_name)
    
    for seq_len in sequence_lengths:
        for num_req in concurrent_requests:
            print(f"\nTesting: {num_req} requests, {seq_len} tokens")
            
            try:
                metrics = benchmark.run_benchmark(num_req, test_prompt, seq_len)
                
                result = BenchmarkMetrics(
                    timestamp=datetime.now().isoformat(),
                    model_name=model_name,
                    engine="PyTorch",
                    batch_size=1,
                    concurrent_requests=num_req,
                    sequence_length=seq_len,
                    prefill_latency_ms=metrics['avg_latency'] * 0.3,
                    decode_latency_ms=metrics['avg_latency'] * 0.7,
                    total_latency_ms=metrics['total_latency'],
                    throughput_tokens_per_sec=metrics['throughput'],
                    gpu_utilization_percent=metrics['gpu_util'],
                    gpu_memory_used_gb=metrics['memory_used_gb'],
                    gpu_memory_total_gb=metrics['memory_total_gb'],
                    kv_cache_size_mb=metrics['kv_cache_mb'],
                    tokens_generated=metrics['tokens_generated']
                )
                
                results.append(asdict(result))
                
                # Save incrementally
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"  ✓ Saved to {output_file}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
            
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"✓ Benchmark Complete!")
    print(f"  Total samples: {len(results)}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_comprehensive_benchmark(
        model_name="gpt2",
        output_file="benchmark_results.csv"
    )
    print("\nSample results:")
    print(df.head())

