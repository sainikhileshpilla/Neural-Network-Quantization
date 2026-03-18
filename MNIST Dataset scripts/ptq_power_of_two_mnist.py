import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from baseline_fp32_train_eval import Net
import os

# Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_len = int(0.9 * len(train_dataset))
val_len = len(train_dataset) - train_len
_, val_data = random_split(train_dataset, [train_len, val_len])
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load FP32 model
model_fp32 = Net()
model_fp32.load_state_dict(torch.load("baseline_model_fp32.pth"))
model_fp32.eval()
print("✅ Loaded FP32 model")

def power_of_two_quantize(weight, n_bits=8):
    w = weight.clone()
    w_min, w_max = w.min(), w.max()
    # Find the closest power-of-two scale
    scale = (w_max - w_min) / (2 ** n_bits - 1)
    pow2_scale = 2 ** torch.round(torch.log2(scale))
    zero_point = torch.round(-w_min / pow2_scale)
    w_q = torch.round(w / pow2_scale + zero_point)
    w_q = torch.clamp(w_q, 0, 2 ** n_bits - 1)
    quantized_weight = pow2_scale * (w_q - zero_point)
    return quantized_weight

# Apply Power-of-Two Quantization to all weights
model_pot = Net()
model_pot.load_state_dict(torch.load("baseline_model_fp32.pth"))
for name, param in model_pot.named_parameters():
    if 'weight' in name:
        param.data = power_of_two_quantize(param.data, n_bits=8)
print(" Applied Power-of-Two Quantization to weights.")

# Evaluate
model_pot.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_pot(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"\n Power-of-Two Quantized Model Test Accuracy: {acc:.2f}%")

torch.save(model_pot.state_dict(), "ptq_power_of_two_model.pth")
print(" Quantized model saved as ptq_power_of_two_model.pth")

# Export quantized model to ONNX for Netron
import torch

dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input shape
onnx_path = "ptq_power_of_two_model.onnx"
torch.onnx.export(model_pot, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
print(f"Exported quantized model to {onnx_path}")

# Model size comparison (in KB)
fp32_size = os.path.getsize("baseline_model_fp32.pth") / 1024
quant_size = os.path.getsize("ptq_power_of_two_model.pth") / 1024
print(f" Model Size - FP32: {fp32_size:.2f} KB | Quantized: {quant_size:.2f} KB")
