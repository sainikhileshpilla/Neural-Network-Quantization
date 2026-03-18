import torch
import os 
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from baseline_fp32_train_eval import Net

# Set quantization backend
torch.backends.quantized.engine = 'fbgemm'

# Load datasets
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

# Split for validation (for tracking QAT)
train_len = int(0.9 * len(train_dataset))
val_len = len(train_dataset) - train_len
train_data, val_data = random_split(train_dataset, [train_len, val_len])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pretrained FP32 model
model = Net()
model.load_state_dict(torch.load("baseline_model_fp32.pth"))
model.train()

print(" Loaded pretrained FP32 model.")

# ------ FUSION (important for QAT) ------
model.fuse_model()
print(" Fused Conv+ReLU modules.")

# Set QAT config (after fusion)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
print(" QAT preparation done (with fusion).")

# Optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# QAT training loop
for epoch in range(5):  # Can increase if needed
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Epoch {epoch+1}: QAT Loss = {total_loss / len(train_loader):.4f}, Val Acc = {100 * correct / total:.2f}%")

# Convert to quantized model (INT8)
model_quantized = torch.quantization.convert(model.eval(), inplace=False)
model_quantized.to('cpu')  # Ensure model is on CPU backend after conversion
print(" Converted to quantized INT8 model.")

# Final test evaluation
model_quantized.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_quantized(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f" QAT Quantized Test Accuracy: {100 * correct / total:.2f}%")

# Save model
torch.save(model_quantized.state_dict(), "qat_model_int8.pth")
print(" Quantized model saved as qat_model_int8.pth")



# Model size comparison (in KB)
fp32_size = os.path.getsize("baseline_model_fp32.pth") / 1024
quant_size = os.path.getsize("qat_model_int8.pth") / 1024
print(f" Model Size - FP32: {fp32_size:.2f} KB | Quantized: {quant_size:.2f} KB")
