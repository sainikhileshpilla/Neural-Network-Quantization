import torch
import torch.quantization
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
print(" Loaded FP32 model")

# Per-channel symmetric quantization config
model_fp32.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.default_observer,
    weight=torch.quantization.default_per_channel_weight_observer
)

# Prepare for calibration
model_prepared = torch.quantization.prepare(model_fp32)
print(" Prepared model for calibration.")

# Calibration
with torch.no_grad():
    for images, _ in val_loader:
        model_prepared(images)
print(" Calibration done.")

# Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)
print(" Converted to per-channel symmetric quantized model.")

# Evaluate
model_quantized.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_quantized(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = 100 * correct / total
print(f"\n Per-channel Symmetric Quantized Model Test Accuracy: {acc:.2f}%")

torch.save(model_quantized.state_dict(), "ptq_per_channel_symmetric_model.pth")
print(" Quantized model saved as ptq_per_channel_symmetric_model.pth")

# Export quantized model to ONNX for Netron
import torch

dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input shape
onnx_path = "ptq_per_channel_symmetric_model.onnx"
torch.onnx.export(model_quantized, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
print(f"Exported quantized model to {onnx_path}")

# Model size comparison (in KB)
fp32_size = os.path.getsize("baseline_model_fp32.pth") / 1024
quant_size = os.path.getsize("ptq_per_channel_symmetric_model.pth") / 1024
print(f" Model Size - FP32: {fp32_size:.2f} KB | Quantized: {quant_size:.2f} KB")
