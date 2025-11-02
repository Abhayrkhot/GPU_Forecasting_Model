#!/usr/bin/env python3
"""
Async Load Tester - Simplified Working Version
"""
import asyncio
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

print("="*80)
print("ASYNC LOAD TESTER STARTING")
print("="*80)

class SimpleAsyncTester:
    def __init__(self, model_name="sshleifer/tiny-gpt2"):
        print(f"\n[1/3] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Model loaded\n")
    
    def generate_once(self, prompt, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return outputs.shape[1] - inputs['input_ids'].shape[1]
    
    async def async_generate(self, request_id, prompt, max_tokens):
        start = time.time()
        loop = asyncio.get_event_loop()
        tokens = await loop.run_in_executor(None, self.generate_once, prompt, max_tokens)
        latency = (time.time() - start) * 1000
        return {'id': request_id, 'latency_ms': latency, 'tokens': tokens}
    
    async def run_test(self, num_requests, prompt, max_tokens=100):
        print(f"[2/3] Testing {num_requests} concurrent requests...")
        
        tasks = [
            self.async_generate(i, prompt, max_tokens)
            for i in range(num_requests)
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        latencies = [r['latency_ms'] for r in results]
        total_tokens = sum(r['tokens'] for r in results)
        
        print(f"\n✓ Test complete!")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Mean latency: {np.mean(latencies):.2f}ms")
        print(f"  P99 latency: {np.percentile(latencies, 99):.2f}ms")
        print(f"  Throughput: {total_tokens/duration:.2f} tokens/sec")
        print(f"  Requests/sec: {num_requests/duration:.2f}\n")
        
        return {
            'num_requests': num_requests,
            'duration': duration,
            'mean_latency': np.mean(latencies),
            'p99_latency': np.percentile(latencies, 99),
            'throughput': total_tokens/duration
        }

async def main():
    print("[Starting Main]\n")
    tester = SimpleAsyncTester()
    
    # Test different concurrency levels
    test_configs = [10, 50, 100, 250, 500, 1000]
    all_results = []
    
    for num_reqs in test_configs:
        result = await tester.run_test(
            num_requests=num_reqs,
            prompt="Explain machine learning",
            max_tokens=50
        )
        all_results.append(result)
        
        # Cool down
        if num_reqs < 1000:
            print("Cooling down 5s...")
            await asyncio.sleep(5)
            torch.cuda.empty_cache()
    
    # Save results
    print("[3/3] Saving results...")
    df = pd.DataFrame(all_results)
    df.to_csv('async_test_results.csv', index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("\n✓ Results saved to async_test_results.csv")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
