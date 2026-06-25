import numpy as np

# Original weights (example tensor)
weights = np.array([
    0.12,
    -0.85,
    1.43,
    -2.10,
    0.57,
    3.20
], dtype=np.float32)

print("Original weights:")
print(weights)
print("dtype:", weights.dtype)
print("Memory:", weights.nbytes, "bytes")

# Quantization (float32 → int8)
max_abs = np.max(np.abs(weights))
scale = max_abs / 127.0

quantized_weights = np.round(weights / scale).astype(np.int8)

print("\nQuantized weights (INT8):")
print(quantized_weights)
print("dtype:", quantized_weights.dtype)
print("Memory:", quantized_weights.nbytes, "bytes")

# Dequantization (int8 → float32 approximation)
dequantized_weights = quantized_weights.astype(np.float32) * scale

print("\nDequantized weights:")
print(dequantized_weights)

# Reconstruction error
error = weights - dequantized_weights

print("\nQuantization error:")
print(error)

print("\nMean absolute error:")
print(np.mean(np.abs(error)))