# EfficientNet‑B0 – Baseline & PTQ Variants (Static, Per‑Channel Symmetric, Power‑of‑Two, AdaRound)

This folder contains five Jupyter notebooks for a **baseline FP32** EfficientNet‑B0 model and four **post‑training quantization (PTQ)** variants:

- `EfficientNet baseline.ipynb` – FP32 training/eval (reference baseline).
- `EfficientNet static.ipynb` – PTQ **static** (per‑tensor, symmetric) calibration.
- `EfficientNet perchannel symmetric.ipynb` – PTQ **per‑channel symmetric** weights.
- `EfficientNet power of two.ipynb` – PTQ **power‑of‑two** weight quantization.
- `EfficientNet adaround.ipynb` – PTQ with **AdaRound** (learned rounding).

Optional pretrained weights are included as `efficientnet_b0_weights.pth` (FP32 `state_dict`).


## 1) Environment

- Python **3.9–3.11** recommended.
- CPU is sufficient for PTQ + eval; CUDA is optional for FP32 training/fine‑tuning.
- PyTorch INT8 uses the **fbgemm** backend on x86 CPUs.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.efficientnet.txt
```

> If you prefer CPU‑only PyTorch, install the CPU wheel for your platform first (see pytorch.org).


## 2) Data layout

All notebooks assume an **ImageFolder** layout (ImageNet‑style) with class subfolders:

```
DATA_ROOT/
  train/
    class_a/
      img1.jpg
      ...
  val/
    class_a/
      ...
  test/             # optional; if missing, val is used for evaluation
    class_a/
    class_b/
```

Set the `DATA_ROOT` / `data_dir` variable near the top of each notebook. Default transforms target **224×224** resolution with ImageNet mean/std.


## 3) Files in this folder

- `EfficientNet baseline.ipynb`
- `EfficientNet static.ipynb`
- `EfficientNet perchannel symmetric.ipynb`
- `EfficientNet power of two.ipynb`
- `EfficientNet adaround.ipynb`
- `efficientnet_b0_weights.pth` (optional baseline weights; load with `model.load_state_dict(torch.load(..., map_location="cpu"))`)
- `requirements.efficientnet.txt`
- `README.efficientnet.md` (this file)


## 4) Running

```bash
jupyter lab     # or: jupyter notebook
```

At the top of each notebook, ensure:
- `DATA_ROOT` / `data_dir` points to your dataset
- `torch.backends.quantized.engine = "fbgemm"` for CPU INT8
- `device = torch.device("cpu")` (recommended for PTQ eval)

Then execute cells in order.


## 5) EfficientNet‑specific notes

- **Swish/SiLU activations:** EfficientNet uses Swish (SiLU). Default PyTorch fusion supports Conv‑BN‑ReLU well; Conv‑BN‑SiLU fusion is limited. Expect **less fusion** than ResNet. This can modestly impact PTQ accuracy/latency.
- **Squeeze‑and‑Excitation (SE) blocks:** Present in MBConv. These are not quantization‑friendly by default; **per‑channel** weight quantization helps.
- **Stochastic depth & dropout:** Make sure `model.eval()` is set for calibration/eval to disable both.
- **Input size:** Use 224×224 and ImageNet mean/std unless your dataset dictates otherwise.
- **Quantization engine:** Prefer `fbgemm` on x86 for best INT8 coverage.


## 6) Workflows

### 6.1 Baseline (FP32)
1. Train/fine‑tune EfficientNet‑B0 (`torchvision.models.efficientnet_b0`).
2. Save FP32 weights: `torch.save(model.state_dict(), "efficientnet_b0_weights.pth")`.
3. Evaluate Top‑1 accuracy on `val` or `test`.

### 6.2 Static PTQ (Per‑Tensor Symmetric)
- Prepare with `torch.ao.quantization` (observer config + any supported fusion).
- Run calibration on a small, representative subset of the training/validation data.
- Convert to a quantized model and evaluate.

### 6.3 Per‑Channel Symmetric PTQ
- Enable **per‑channel** quantization for conv weights; tends to help with MBConv blocks.

### 6.4 Power‑of‑Two (PoT)
- Constrain weight scales to powers of two for simpler integer shifts in deployment.
- Compare accuracy/latency/size trade‑offs to static/per‑channel.

### 6.5 AdaRound
- Perform learned rounding on weights with a calibration/rounding set.
- Export the rounded (quantized) model and evaluate.


## 7) Measuring accuracy, size, latency

**Accuracy (Top‑1):** Each notebook provides an evaluation cell on `val`/`test`.

**Model size:**

```python
torch.save(model.state_dict(), "model_quant.pth")
import os; print(os.path.getsize("model_quant.pth")/1e6, "MB")
```

**CPU latency:** (warm up before timing)

```python
import time, torch
model.eval()
example = torch.randn(1, 3, 224, 224)
with torch.inference_mode():
    for _ in range(20): _ = model(example)   # warmup
    t0 = time.time()
    iters = 200
    for _ in range(iters): _ = model(example)
    print((time.time()-t0)/iters*1000, "ms/img")
```

Tips: use `torch.inference_mode()`, disable gradients, and run on CPU for fair PTQ comparisons.


## 8) Reproducibility

- Seeds: `torch.manual_seed(42)` and `np.random.seed(42)` where provided.
- Use consistent transforms and evaluation splits across methods.
- Keep the same calibration subset across all PTQ variants.


## 9) Converting notebooks to scripts (optional)

```bash
jupyter nbconvert --to script "EfficientNet baseline.ipynb"
jupyter nbconvert --to script "EfficientNet static.ipynb"
jupyter nbconvert --to script "EfficientNet perchannel symmetric.ipynb"
jupyter nbconvert --to script "EfficientNet power of two.ipynb"
jupyter nbconvert --to script "EfficientNet adaround.ipynb"
```

Edit paths in the generated `.py` files and run with `python <script>.py`.


## 10) Troubleshooting

- **Torch/torchvision mismatch** → install matching versions (see PyTorch install matrix).
- **No `fbgemm`** on your CPU → INT8 speedups may be limited; try `qnnpack` or stick to FP32.
- **OOM** → reduce batch size; keep input size at 224×224; avoid unnecessary copies.
- **Accuracy drop after PTQ** → enlarge calibration set; use per‑channel or AdaRound; be aware of limited Conv‑BN‑SiLU fusion; verify SE blocks’ observers.


## 11) References

- **EfficientNet**: Tan & Le, *ICML 2019*.
- **PyTorch Quantization**: `torch.ao.quantization`.
- **AdaRound**: Bannerjee et al., *ICLR 2020*.
