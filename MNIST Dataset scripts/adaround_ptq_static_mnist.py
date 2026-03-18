import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from baseline_fp32_train_eval import Net
import os
import copy
from tqdm import tqdm

# Load pre-trained FP32 model
model = Net()
model.load_state_dict(torch.load("baseline_model_fp32.pth"))
model.eval()
print("✅ Loaded FP32 model")

# Load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform)

train_len = int(0.9 * len(train_dataset))
val_len = len(train_dataset) - train_len
_, val_data = random_split(train_dataset, [train_len, val_len])

val_loader  = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# AdaRound Function

def adaround_weight_tensor(weight, n_bits=8, num_iters=1000, lr=1e-2):
    # AdaRound: Learnable rounding offsets for quantization
    # weight: torch.Tensor (the weight to quantize)
    # n_bits: int (number of bits for quantization)
    # num_iters: int (number of optimization steps)
    # lr: float (learning rate)
    w = weight.clone()
    scale = (w.max() - w.min()) / (2 ** n_bits - 1)
    zero_point = torch.round(-w.min() / scale)
    alpha = torch.zeros_like(w, requires_grad=True)  # Use tensor with requires_grad
    optimizer = torch.optim.Adam([alpha], lr=lr)
    for i in range(num_iters):
        optimizer.zero_grad()
        soft_offset = torch.sigmoid(alpha)
        w_q = torch.floor(w / scale) + soft_offset
        soft_weight = w_q * scale
        # Ensure loss is a scalar for backward()
        loss = torch.nn.functional.mse_loss(soft_weight, w)
        loss.backward(retain_graph=True)  # No detach, no retain_graph
        optimizer.step()
        # Re-create alpha as a new leaf tensor for next iteration
        alpha = alpha.detach().requires_grad_()
        if i == 0 or i == num_iters - 1:
            print(f"Iter {i}: loss={loss.item():.6f}, alpha.requires_grad={alpha.requires_grad}, soft_weight.requires_grad={soft_weight.requires_grad}, loss.requires_grad={loss.requires_grad}, loss.grad_fn={loss.grad_fn}")
    with torch.no_grad():
        hard_offset = (torch.sigmoid(alpha) > 0.5).float()
        w_q = torch.floor(w / scale) + hard_offset
        quantized_weight = w_q * scale
    return quantized_weight


# Apply AdaRound to Conv2d weights
def apply_adaround(model, n_bits=8, num_iters=1000, lr=1e-2):
    model_adaround = copy.deepcopy(model)
    model_adaround.eval()
    # Remove torch.no_grad() to allow gradients
    for images, _ in val_loader:
        # Step 1: Forward through conv1 + relu + pool1
        x = model_adaround.pool1(model_adaround.relu1(model_adaround.conv1(images)))

        # Step 2: AdaRound conv2 using output of conv1 block
        # Do NOT use .data for assignment during optimization
        model_adaround.conv2.weight = nn.Parameter(adaround_weight_tensor(
            model_adaround.conv2.weight,
            n_bits=n_bits,
            num_iters=num_iters,
            lr=lr
        ))
        break  # Only one batch needed for rounding

    return model_adaround


# Perform AdaRound
model_adaround = apply_adaround(model, n_bits=8, num_iters=1000, lr=1e-2)

# Evaluate AdaRounded model
model_adaround.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_adaround(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n AdaRound Quantized Accuracy: {accuracy:.2f}%")

# Save
torch.save(model_adaround.state_dict(), "ptq_static_model_adaround.pth")
print(" Saved AdaRounded model: ptq_static_model_adaround.pth")

# Compare sizes
fp32_size = os.path.getsize("baseline_model_fp32.pth") / 1024
quant_size = os.path.getsize("ptq_static_model_adaround.pth") / 1024
print(f" Model Size - FP32: {fp32_size:.2f} KB | AdaRound Quantized: {quant_size:.2f} KB")
