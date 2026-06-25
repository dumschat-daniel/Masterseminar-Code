import torch
import torch.nn as nn
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type != "cuda":
    print("\nWARNING:")
    print("CUDA GPU not detected.")
    print("FP16 may not provide real speedups on CPU.")
    print("Use an NVIDIA GPU for meaningful FP16 benchmarks.")


class SimpleMLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=4096, output_size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def model_memory_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def activation_memory_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def benchmark(model, input_tensor, runs=100):
    with torch.no_grad():
        for _ in range(50):  # warm-up
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    return ((end - start) / runs) * 1000  # ms per run


# Models
model_fp32 = SimpleMLP().to(device).eval()
model_fp16 = SimpleMLP().half().to(device).eval()

# Inputs
input_fp32 = torch.randn(64, 2048, device=device, dtype=torch.float32)
input_fp16 = input_fp32.half()

# Memory usage
fp32_model_mem = model_memory_bytes(model_fp32)
fp16_model_mem = model_memory_bytes(model_fp16)

# Forward pass (for activation size)
with torch.no_grad():
    out_fp32 = model_fp32(input_fp32)
    out_fp16 = model_fp16(input_fp16)

fp32_activation_mem = activation_memory_bytes(out_fp32)
fp16_activation_mem = activation_memory_bytes(out_fp16)

# Latency
fp32_latency = benchmark(model_fp32, input_fp32)
fp16_latency = benchmark(model_fp16, input_fp16)

# Results
print("\n" + "=" * 70)
print("NEURAL NETWORK PRECISION COMPARISON")
print("=" * 70)

print("\nModel Memory:")
print(f"FP32: {fp32_model_mem / (1024**2):.2f} MB")
print(f"FP16: {fp16_model_mem / (1024**2):.2f} MB")

print("\nActivation Memory:")
print(f"FP32: {fp32_activation_mem / 1024:.2f} KB")
print(f"FP16: {fp16_activation_mem / 1024:.2f} KB")

print("\nLatency:")
print(f"FP32: {fp32_latency:.3f} ms")
print(f"FP16: {fp16_latency:.3f} ms")

print("\nReduction:")
print(f"Model memory: {fp32_model_mem / fp16_model_mem:.2f}x")
print(f"Activation:   {fp32_activation_mem / fp16_activation_mem:.2f}x")
print(f"Speedup:      {fp32_latency / fp16_latency:.2f}x")