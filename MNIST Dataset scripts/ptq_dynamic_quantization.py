import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from baseline_model import Net  # Make sure Net is defined properly
import os

# Device configuration
device = torch.device('cpu')

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load FP32 model
model_fp32 = Net().to(device)
model_fp32.load_state_dict(torch.load("baseline_model_fp32.pth", map_location=device))
model_fp32.eval()
print("Loaded full precision (FP32) model.")

# Apply dynamic quantization
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model_fp32,
    {nn.Linear},
    dtype=torch.qint8
)
print("Applied dynamic quantization to Linear layers.")

# Evaluate accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_dynamic_quantized(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Dynamic Quantized Model Test Accuracy: {accuracy:.2f}%")

# Save quantized model
quantized_model_path = "ptq_dynamic_model.pth"  # << Changed filename
torch.save(model_dynamic_quantized.state_dict(), quantized_model_path)
print(f"Quantized model saved as {quantized_model_path}")

# Export quantized model to ONNX for Netron
#import torch

#dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input shape
#onnx_path = "ptq_dynamic_model.onnx"
#torch.onnx.export(model_dynamic_quantized, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
#print(f"Exported quantized model to {onnx_path}")

# Compare model sizes
fp32_size = os.path.getsize("baseline_model_fp32.pth") / 1024
quant_size = os.path.getsize(quantized_model_path) / 1024
print(f"Model Size - FP32: {fp32_size:.2f} KB | Quantized: {quant_size:.2f} KB")

