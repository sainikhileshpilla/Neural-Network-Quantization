# DenseNet‑121 – Baseline & PTQ Variants (Static, Per‑Channel Symmetric, Power‑of‑Two, AdaRound)

This folder contains five Jupyter notebooks for a **baseline FP32** DenseNet‑121 model and four **post‑training quantization (PTQ)** variants:

- `DenseNet121 baseline.ipynb` – FP32 training/eval (reference baseline).
- `DenseNet121 static.ipynb` – PTQ **static** (per‑tensor, symmetric) calibration.
- `DenseNet121 perchannel symmetric.ipynb` – PTQ **per‑channel symmetric** weights.
- `DesneNet121 power of two.ipynb` – PTQ **power‑of‑two** weight quantization. *(Note: file name has a typo in "DesneNet121" on purpose to match the provided notebook.)*
- `DenseNet121 adaround.ipynb` – PTQ with **AdaRound** (learned rounding).

Optional pretrained weights are included as `densenet121_weights.pth` (FP32 `state_dict`).


## 1) Environment

- Python **3.9–3.11** recommended.
- CPU is sufficient for PTQ + eval; CUDA is optional for FP32 training/fine‑tuning.
- PyTorch INT8 uses the **fbgemm** backend on x86 CPUs.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.densenet121.txt
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

- `DenseNet121 baseline.ipynb`
- `DenseNet121 static.ipynb`
- `DenseNet121 perchannel symmetric.ipynb`
- `DesneNet121 power of two.ipynb`
- `DenseNet121 adaround.ipynb`
- `densenet121_weights.pth` (optional baseline weights; load with `model.load_state_dict(torch.load(..., map_location="cpu"))`)
- `requirements.densenet121.txt`
- `README.densenet121.md` (this file)


## 4) Running

```bash
jupyter lab     # or: jupyter notebook
```

At the top of each notebook, ensure:
- `DATA_ROOT` / `data_dir` points to your dataset
- `torch.backends.quantized.engine = "fbgemm"` for CPU INT8
- `device = torch.device("cpu")` (recommended for PTQ eval)

Then execute cells in order.


## 5) DenseNet‑specific notes

- **Heavy BatchNorm usage:** DenseNet relies on many BN layers across dense blocks. Not all Conv‑BN‑ReLU patterns are fusible due to the concatenations across layers. Expect **limited fusion**, which can impact PTQ performance.
- **Feature concatenation:** Dense connections expand channel dimensions; ensure calibration batches cover typical activation ranges to avoid quantization outliers.
- **Per‑channel quantization:** Often helps maintain accuracy on conv weights given the varying scales introduced by concatenation.
- **Dropout / eval mode:** Use `model.eval()` for calibration and evaluation to disable dropout and use running BN stats.
- **Input size:** 224×224, ImageNet mean/std normalization.


## 6) Workflows

### 6.1 Baseline (FP32)
1. Train/fine‑tune DenseNet‑121 (`torchvision.models.densenet121`).
2. Save FP32 weights: `torch.save(model.state_dict(), "densenet121_weights.pth")`.
3. Evaluate Top‑1 accuracy on `val` or `test`.

### 6.2 Static PTQ (Per‑Tensor Symmetric)
- Prepare with `torch.ao.quantization` (observer config + any supported fusion).
- Run calibration on a small, representative subset of the training/validation data.
- Convert to a quantized model and evaluate.

### 6.3 Per‑Channel Symmetric PTQ
- Enable **per‑channel** quantization for conv weights; tends to help with DenseNet’s variable channel scales.

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
jupyter nbconvert --to script "DenseNet121 baseline.ipynb"
jupyter nbconvert --to script "DenseNet121 static.ipynb"
jupyter nbconvert --to script "DenseNet121 perchannel symmetric.ipynb"
jupyter nbconvert --to script "DesneNet121 power of two.ipynb"
jupyter nbconvert --to script "DenseNet121 adaround.ipynb"
```

Edit paths in the generated `.py` files and run with `python <script>.py`.


## 10) Troubleshooting

- **Torch/torchvision mismatch** → install matching versions (see PyTorch install matrix).
- **No `fbgemm`** on your CPU → INT8 speedups may be limited; try `qnnpack` or stick to FP32.
- **OOM** → reduce batch size; keep input size at 224×224; avoid unnecessary copies.
- **Accuracy drop after PTQ** → enlarge calibration set; prefer per‑channel; check BN folding/fusion feasibility in dense blocks.


## 11) References

- **DenseNet**: Huang et al., *CVPR 2017*.
- **PyTorch Quantization**: `torch.ao.quantization`.
- **AdaRound**: Bannerjee et al., *ICLR 2020*.
