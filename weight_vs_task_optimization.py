import itertools
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# Synthetic task: nonlinear interaction between parameters
# y = sigmoid(w1 * x) * w2
N = 2000
X = torch.randn(N, 1) * 2

W1_true, W2_true = 3.0, 5.0
y = torch.sigmoid(W1_true * X.squeeze()) * W2_true + 0.05 * torch.randn(N)

X_train, y_train = X[:1600], y[:1600]
X_test, y_test = X[1600:], y[1600:]


class ScaledSigmoid(nn.Module):
    def __init__(self, w1=None, w2=None):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([1.0 if w1 is None else w1]))
        self.w2 = nn.Parameter(torch.tensor([1.0 if w2 is None else w2]))

    def forward(self, x):
        return torch.sigmoid(self.w1 * x.squeeze()) * self.w2


# Fit continuous model
model = ScaledSigmoid()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for _ in range(1000):
    loss = nn.MSELoss()(model(X_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

w1_orig = model.w1.item()
w2_orig = model.w2.item()

print(f"Learned: w1={w1_orig:.4f}, w2={w2_orig:.4f}")


# Discrete grid (quantized parameter candidates)
grid_w1 = [2.0, 2.5, 3.5, 4.0]
grid_w2 = [4.0, 4.5, 5.5, 6.0]


def eval_mse(w1, w2):
    m = ScaledSigmoid(w1, w2)
    with torch.no_grad():
        return nn.MSELoss()(m(X_test), y_test).item()


# ── Objective 1: parameter-space distance ───────────────────────────────────
best_param = float("inf")
best_W_param = None

for w1, w2 in itertools.product(grid_w1, grid_w2):
    d = (w1 - w1_orig) ** 2 + (w2 - w2_orig) ** 2
    if d < best_param:
        best_param = d
        best_W_param = (w1, w2)


# ── Objective 2: task loss ───────────────────────────────────────────────────
best_task = float("inf")
best_W_task = None

for w1, w2 in itertools.product(grid_w1, grid_w2):
    mse = eval_mse(w1, w2)
    if mse < best_task:
        best_task = mse
        best_W_task = (w1, w2)


# ── Full grid report ─────────────────────────────────────────────────────────
print("\nGrid search results")
print(f"{'w1':>5} {'w2':>5} {'||ΔW||²':>10} {'MSE':>10}")
print("-" * 40)

for w1, w2 in itertools.product(grid_w1, grid_w2):
    d = (w1 - w1_orig) ** 2 + (w2 - w2_orig) ** 2
    mse = eval_mse(w1, w2)

    tag = ""
    if (w1, w2) == best_W_param and (w1, w2) == best_W_task:
        tag = "  ← both"
    elif (w1, w2) == best_W_param:
        tag = "  ← param-opt"
    elif (w1, w2) == best_W_task:
        tag = "  ← task-opt"

    print(f"{w1:>5.1f} {w2:>5.1f} {d:>10.4f} {mse:>10.4f}{tag}")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Method':<20} {'w1':>5} {'w2':>5} {'||ΔW||²':>10} {'MSE':>10}")
print("=" * 60)

for label, (w1, w2) in [
    ("Original (fp32)", (w1_orig, w2_orig)),
    ("Param-space", best_W_param),
    ("Task-loss", best_W_task),
]:
    d = (w1 - w1_orig) ** 2 + (w2 - w2_orig) ** 2
    mse = eval_mse(w1, w2)
    print(f"{label:<20} {w1:>5.2f} {w2:>5.2f} {d:>10.4f} {mse:>10.4f}")

print("=" * 60)

# Comparison insight
if best_W_param != best_W_task:
    p_mse = eval_mse(*best_W_param)
    t_mse = eval_mse(*best_W_task)

    print(f"\nParam-space choice: {best_W_param} → MSE {p_mse:.4f}")
    print(f"Task-opt choice:     {best_W_task} → MSE {t_mse:.4f}")
    print(f"Task-opt is {p_mse / t_mse:.2f}x better on the real objective.")
else:
    print("\nBoth methods agree — adjust grid to see divergence.")