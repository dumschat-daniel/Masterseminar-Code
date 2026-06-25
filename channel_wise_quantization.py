import numpy as np

# Example channels with different value ranges
channel_small = np.array([0.01, 0.05, 0.08, 0.12])
channel_large = np.array([5.0, 8.0, 12.0, 15.0])

weights = np.stack([channel_small, channel_large])

BITS = 4
QMAX = 2**BITS - 1

# Global quantization
global_min = weights.min()
global_max = weights.max()

global_scale = (global_max - global_min) / QMAX

global_quantized = np.round(
    (weights - global_min) / global_scale
)

global_dequantized = (
    global_quantized * global_scale + global_min
)

print("GLOBAL QUANTIZATION")
print(global_dequantized)

# Per-channel quantization
per_channel_dequantized = []

for channel in weights:
    cmin = channel.min()
    cmax = channel.max()

    scale = (cmax - cmin) / QMAX

    q = np.round((channel - cmin) / scale)
    dq = q * scale + cmin

    per_channel_dequantized.append(dq)

per_channel_dequantized = np.array(per_channel_dequantized)

print("\nPER-CHANNEL QUANTIZATION")
print(per_channel_dequantized)