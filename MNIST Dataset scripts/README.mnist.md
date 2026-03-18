# MNIST CNN – FP32, PTQ (Static/Per‑Channel/PoT/Dynamic/AdaRound) & QAT

This repo contains a simple CNN for **MNIST** and multiple quantization workflows:
- **Baseline FP32** training/eval → produces a reference checkpoint.
- **PTQ Static** (per‑tensor, symmetric)
- **PTQ Per‑Channel Symmetric**
- **PTQ Power‑of‑Two (PoT)**
- **PTQ Dynamic** (Linear layers)
- **PTQ AdaRound** (learned rounding of weights)
- **QAT** (Quantization‑Aware Training)

> All scripts run on CPU and download MNIST automatically.


## 1) Files & what they do

- `baseline_model_withoutquant_withoutdequant.py` – trains a small CNN on MNIST and saves **FP32** weights as `models/baseline_model_fp32.pth`. It defines the `Net` class and includes a simple `fuse_model` helper (Conv+ReLU).
- `ptq_static_quantization.py` – static PTQ using `fbgemm`: prepare → calibrate on a val split → convert. Exports ONNX and saves `ptq_static_model.pth`.
- `ptq_per_channel_symmetric_mnist.py` – per‑channel weight observers for conv layers, then prepare/convert. Exports ONNX and saves `ptq_per_channel_symmetric_model.pth`.
- `ptq_power_of_two_mnist.py` – custom power‑of‑two quantizer for **weights**, evaluate, export ONNX, save `ptq_power_of_two_model.pth`.
- `ptq_dynamic_quantization.py` – dynamic quantization of `nn.Linear` with `quantize_dynamic`, saves `ptq_dynamic_model.pth`. (Imports `Net` from baseline; adjust import if needed.)
- `adaround_ptq_static_mnist.py` – applies **AdaRound** (learned rounding) to conv weights using a small val set; evaluates and saves `ptq_static_model_adaround.pth`.
- `qat_model.py` – Quantization‑Aware Training with fusion → `prepare_qat` → short training → `convert` to INT8, saves `qat_model_int8.pth`.


## 2) Environment

- Python **3.9–3.11** recommended.
- CPU backend **fbgemm** is used for INT8 ops on x86 (set by scripts where relevant).
- CUDA is **not required** for MNIST; all flows run on CPU.

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.mnist.txt
```


## 3) Dataset

Scripts auto‑download MNIST to `./data`. Transforms are `ToTensor()`; some scripts also **normalize** with MNIST mean/std.


## 4) Quickstart

1) **Train baseline FP32**

```bash
python baseline_model_withoutquant_withoutdequant.py
# -> saves models/baseline_model_fp32.pth
```

2) **Unify checkpoint path** (some PTQ scripts expect the file at repo root):

```bash
# Option A: copy to root
cp models/baseline_model_fp32.pth baseline_model_fp32.pth
# Option B: edit the scripts to load from models/baseline_model_fp32.pth
```

3) **Run a PTQ/QAT variant**, e.g.:

```bash
python ptq_static_quantization.py
python ptq_per_channel_symmetric_mnist.py
python ptq_power_of_two_mnist.py
python ptq_dynamic_quantization.py
python adaround_ptq_static_mnist.py
python qat_model.py
```


## 5) Notes & gotchas

- **Imports / filenames:** If a script imports `Net` from a different name, change it to `from baseline_model_withoutquant_withoutdequant import Net` (or make a tiny `baseline_model.py` wrapper).
- **ONNX export:** Several PTQ scripts export ONNX (`*.onnx`) for tools like Netron—comment out if not needed.
- **Fusion:** Static/QAT flows try to fuse Conv+ReLU; QAT requires fusion **before** `prepare_qat`.
- **Calibration:** Static/per‑channel PTQ use a small held‑out val subset; more calibration usually helps.
- **AdaRound:** Use a reasonably sized rounding set for stability.
- **Latency:** Measure with `torch.inference_mode()` and average many iterations.


## 6) License & citation

- Educational benchmarking of quantization techniques on MNIST.
- Cite PyTorch quantization docs and AdaRound if you publish results.
