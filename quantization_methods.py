import numpy as np
import matplotlib.pyplot as plt
import time

# Synthetic activation distribution
# Most values are near zero with a few large outliers
np.random.seed(42)

main_distribution = np.random.normal(0.0, 1.0, 100000)
outliers = np.random.normal(0.0, 12.0, 300)

activations = np.concatenate([main_distribution, outliers])

# ------------------------------------------------------------
# Quantization helper
# ------------------------------------------------------------
def quantize(values, min_val, max_val, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    clipped = np.clip(values, min_val, max_val)

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    quantized = np.round(clipped / scale + zero_point)
    dequantized = (quantized - zero_point) * scale

    mse = np.mean((values - dequantized) ** 2)

    return {
        "dequantized": dequantized,
        "scale": scale,
        "mse": mse,
        "range": (min_val, max_val),
    }

# ------------------------------------------------------------
# 1. Min-max calibration
# ------------------------------------------------------------
start = time.perf_counter()

minmax_min = activations.min()
minmax_max = activations.max()

minmax_result = quantize(activations, minmax_min, minmax_max)

minmax_time = time.perf_counter() - start

# ------------------------------------------------------------
# 2. Percentile clipping
# ------------------------------------------------------------
start = time.perf_counter()

percentile_min = np.percentile(activations, 0.1)
percentile_max = np.percentile(activations, 99.9)

percentile_result = quantize(activations, percentile_min, percentile_max)

percentile_time = time.perf_counter() - start

# ------------------------------------------------------------
# 3. Threshold search (entropy-inspired)
# ------------------------------------------------------------
start = time.perf_counter()

candidate_thresholds = np.linspace(2.0, 10.0, 50)

best_threshold = None
best_result = None
best_mse = float("inf")

for t in candidate_thresholds:
    result = quantize(activations, -t, t)

    if result["mse"] < best_mse:
        best_mse = result["mse"]
        best_threshold = t
        best_result = result

entropy_result = best_result
entropy_time = time.perf_counter() - start

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
print("\nQuantization Calibration Comparison")
print("=" * 60)

print("\nMin-Max")
print("-" * 40)
print(f"Range: [{minmax_result['range'][0]:.2f}, {minmax_result['range'][1]:.2f}]")
print(f"Scale: {minmax_result['scale']:.4f}")
print(f"MSE:   {minmax_result['mse']:.4f}")
print(f"Time:  {minmax_time:.6f}s")

print("\nPercentile Clipping")
print("-" * 40)
print(f"Range: [{percentile_result['range'][0]:.2f}, {percentile_result['range'][1]:.2f}]")
print(f"Scale: {percentile_result['scale']:.4f}")
print(f"MSE:   {percentile_result['mse']:.4f}")
print(f"Time:  {percentile_time:.6f}s")

print("\nEntropy-Inspired")
print("-" * 40)
print(f"Range: [{entropy_result['range'][0]:.2f}, {entropy_result['range'][1]:.2f}]")
print(f"Scale: {entropy_result['scale']:.4f}")
print(f"MSE:   {entropy_result['mse']:.4f}")
print(f"Time:  {entropy_time:.6f}s")

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.hist(activations, bins=200, density=True, alpha=0.7,
         label="Activations")

# Min-max
plt.axvline(minmax_min, linestyle="--")
plt.axvline(minmax_max, linestyle="--", label="Min-Max")

# Percentiles
plt.axvline(percentile_min, linestyle=":")
plt.axvline(percentile_max, linestyle=":", label="Percentile")

# Entropy-inspired
plt.axvline(-best_threshold, linestyle="-.")
plt.axvline(best_threshold, linestyle="-.", label="Entropy-inspired")

plt.title("Calibration Strategies for Post-Training Quantization")
plt.xlabel("Activation Value")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()
