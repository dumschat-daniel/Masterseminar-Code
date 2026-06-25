import torch
import torchvision.models as models

def report(name, model):
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    print(f"{name:<15} {n_params:>12,} params    {size_mb:>8.2f} MB (FP32)")

models_list = [
    ("ResNet-18", models.resnet18(weights=None)),
    ("ResNet-50", models.resnet50(weights=None)),
    ("MobileNetV2", models.mobilenet_v2(weights=None)),
]

print(f"{'Model':<15} {'Parameters':>18}    {'Size':>12}")
print("-" * 50)

for name, model in models_list:
    report(name, model)