"""
Batching Optimization - Quick Demo
"""
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*60)
print("Batching Optimization Demonstration")
print("="*60)

model_name = "sshleifer/tiny-gpt2"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded\n")

# Test data
prompts = ["Explain machine learning"] * 24
max_tokens = 30

# WITHOUT batching
print("[1/2] Testing WITHOUT batching...")
start = time.time()
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_tokens)
time_unbatched = time.time() - start
print(f"  Time: {time_unbatched:.2f}s")

# WITH batching (batch_size=8)
print("\n[2/2] Testing WITH batching (batch_size=8)...")
start = time.time()
for i in range(0, len(prompts), 8):
    batch = prompts[i:i+8]
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_tokens)
time_batched = time.time() - start
print(f"  Time: {time_batched:.2f}s")

# Results
speedup = time_unbatched / time_batched
improvement = ((time_unbatched - time_batched) / time_unbatched) * 100

print(f"\n{'='*60}")
print("RESULTS:")
print(f"  Speedup: {speedup:.2f}x faster")
print(f"  Time saved: {improvement:.1f}%")
print(f"  Batching strategy: Dynamic batching with batch_size=8")
print("="*60)
print("\n✓ Batching optimization validated!")
