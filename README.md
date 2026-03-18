# Master Thesis – Post‑Training Quantization & QAT (CMS)

This repository contains experiments and scripts for evaluating **quantization** techniques across multiple CNN architectures and a small MNIST model. It includes:
- **ImageNet‑style models (torchvision)**: ShuffleNet‑V2, ResNet‑50, GoogLeNet (Inception‑V1), EfficientNet‑B0, DenseNet‑121
- **MNIST CNN**: FP32 baseline, PTQ variants (static, per‑channel, power‑of‑two, dynamic, AdaRound) and QAT

All PTQ flows use **PyTorch `torch.ao.quantization`** unless noted. Where applicable, CPU INT8 runs rely on the **fbgemm** backend.


## 1) Repository layout

```
.
├─ notebooks/
│  ├─ ShuffleNet/
│  ├─ ResNet50/
│  ├─ GoogLeNet/
│  ├─ EfficientNet/
│  └─ DenseNet121/
├─ scripts_mnist/
├─ weights/
├─ results/
└─ requirements-all.txt
```
> Your actual file structure may vary slightly; adapt paths in notebooks where needed.


## 2) Environment & installation

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements-all.txt
```
- **CPU‑only**: install the CPU wheel for PyTorch/torchvision that matches your OS/Arch before the rest.
- **CUDA**: install the matching PyTorch/torchvision build from pytorch.org (e.g., cu121).
- Launch notebooks: `jupyter lab`


## 3) Datasets

### 3.1 Tiny‑ImageNet‑200 (dataset used)
This project trains/evaluates the ImageNet‑style models on **Tiny‑ImageNet‑200** (64×64 images, 200 classes).  
**Source (Kaggle):** https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

**Download via Kaggle CLI:**
```bash
# requires: pip install kaggle  &&  kaggle.json token configured
mkdir -p data && cd data
kaggle datasets download -d akash2sharma/tiny-imagenet -p . --unzip
# results in: ./tiny-imagenet-200/...
```

**Folder structure (as provided):**
```
tiny-imagenet-200/
  train/
    <wnid>/
      images/
      boxes.txt         # may be empty/unused
  val/
    images/
    val_annotations.txt
  test/
    images/
  wnids.txt
  words.txt
```

**Optional – reformat `val/` into class subfolders (ImageFolder‑style):**
```bash
cd tiny-imagenet-200
mkdir -p val_split
while read IMG WNID _; do
  mkdir -p val_split/$WNID
  mv val/images/$IMG val_split/$WNID/
done < val/val_annotations.txt
# now you can use: train/ and val_split/ as ImageFolder splits
```

**Set this path** near the top of each notebook:
```python
DATA_ROOT = "/path/to/tiny-imagenet-200"
VAL_DIR   = f"{DATA_ROOT}/val_split"  # if you applied the optional split
```

> If you prefer to resize to the canonical 224×224 expected by many torchvision models,
> keep the transforms as `Resize(256) -> CenterCrop(224)` (or direct `Resize((224,224))`).
> The networks will upscale from 64×64 to 224×224. Alternatively, you can **change the
> transforms** to 64×64 and fine‑tune the first layers accordingly—results will differ.


### 3.2 MNIST
MNIST scripts auto‑download to `./data`.


## 4) How to run

### 4.1 ImageNet‑style models (notebooks)
1. Open a model notebook (e.g., `notebooks/ResNet50/ResNet-50 static.ipynb`).
2. Set:
   ```python
   import torch
   torch.backends.quantized.engine = "fbgemm"   # x86 CPU
   DATA_ROOT = "/path/to/tiny-imagenet-200"
   VAL_DIR   = f"{DATA_ROOT}/val_split"  # if you created it
   device    = torch.device("cpu")       # PTQ eval on CPU
   ```
3. Run cells in order: Baseline → Prepare/Calibrate/Convert → Eval/Latency/Size.

### 4.2 MNIST scripts
```bash
cd scripts_mnist
python baseline_model_withoutquant_withoutdequant.py
python ptq_static_quantization.py
python ptq_per_channel_symmetric_mnist.py
python ptq_power_of_two_mnist.py
python ptq_dynamic_quantization.py
python adaround_ptq_static_mnist.py
python qat_model.py
```
Some scripts expect baseline at repo root; copy from `models/` or edit the load path.


## 5) Implemented methods

- Static PTQ (per‑tensor symmetric)
- Per‑channel symmetric PTQ
- Power‑of‑Two PTQ (weights)
- Dynamic quantization (Linear)
- AdaRound (learned rounding)
- QAT (fuse → prepare_qat → short train → convert)


## 6) Metrics & logging

- **Top‑1 accuracy**: evaluation cells/scripts.
- **Model size**: compare `*.pth` sizes.
- **Latency**:
  ```python
  import time, torch
  model.eval(); ex = torch.randn(1, 3, 224, 224)
  with torch.inference_mode():
      for _ in range(20): _ = model(ex)
      t0 = time.time(); it = 200
      for _ in range(it): _ = model(ex)
      print((time.time()-t0)/it*1000, "ms/img")
  ```


## 7) Reproducibility

- Fix seeds (`torch.manual_seed(42)`, `np.random.seed(42)`).
- Keep transforms and calibration subsets consistent across methods.
- Evaluate on the same split.


## 8) Caveats

- Fusion coverage differs (EfficientNet Conv‑BN‑SiLU, DenseNet dense connections).
- GoogLeNet: disable **aux logits** for quantization/eval.
- INT8 speedups vary on non‑x86 (fallback to `qnnpack` or FP32).
- Training at 64×64 vs. upscaling to 224×224 will change accuracy; pick one policy and keep it consistent across methods.


## 9) References

ResNet (He16), GoogLeNet (Szegedy15), ShuffleNet‑V2 (Ma18), EfficientNet (Tan&Le19), DenseNet (Huang17); PyTorch Quantization (`torch.ao.quantization`); AdaRound (Bannerjee20).


## 10) License

Educational research for a master’s thesis at TU Dresden (CMS).
