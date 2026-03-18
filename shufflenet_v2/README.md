# ShuffleNet V2 – Baseline & PTQ Variants (Static, Per‑Channel Symmetric, Power‑of‑Two, AdaRound)

This folder contains five Jupyter notebooks implementing a **baseline FP32** ShuffleNet V2 model and four **post‑training quantization (PTQ)** variants:

- `Shufflenet V2 baseline.ipynb` – FP32 training/eval (reference baseline).
- `Shufflenet static.ipynb` – standard PTQ **static** (per‑tensor, symmetric) calibration.
- `Shufflenet perchannel symmetric.ipynb` – PTQ **per‑channel symmetric** weights.
- `Shufflenet power of two.ipynb` – PTQ **power‑of‑two** weight quantization.
- `Shufflenet adaround.ipynb` – PTQ with **AdaRound** (learned rounding).

Optional pretrained weights are included as `shufflenetv2_weights.pth` (state_dict).


## 1) Environment

- Python **3.9–3.11** recommended.
- CPU is sufficient (PTQ + eval). CUDA is optional for the FP32 baseline training step.
- PyTorch with the `fbgemm` backend is used for INT8 ops on CPU.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer CPU‑only PyTorch, install the CPU wheel for your platform (see PyTorch site) before the rest of the requirements.


## 2) Data layout

All notebooks expect an **ImageFolder**‑style dataset (e.g., ImageNet‑like) with class subfolders:

```
DATA_ROOT/
  train/
    class_a/
      img1.jpg
      ...
    class_b/
      ...
  val/
    class_a/
      ...
    class_b/
      ...
  test/           # optional; if missing, val is used for evaluation
    class_a/
    class_b/
```

Update the `data_dir` or `DATA_ROOT` variable at the top of each notebook to point to your dataset.


## 3) Files in this folder

- `Shufflenet V2 baseline.ipynb`
- `Shufflenet static.ipynb`
- `Shufflenet perchannel symmetric.ipynb`
- `Shufflenet power of two.ipynb`
- `Shufflenet adaround.ipynb`
- `shufflenetv2_weights.pth` (optional baseline weights; load with `model.load_state_dict(torch.load(..., map_location="cpu"))`)
- `requirements.txt`
- `README.md` (this file)


## 4) Running the notebooks

Launch Jupyter and open any notebook:

```bash
jupyter lab    # or: jupyter notebook
```

At the top of each notebook, set:
- `DATA_ROOT` (or `data_dir`) to your dataset
- `torch.backends.quantized.engine = "fbgemm"` for CPU INT8
- `device = torch.device("cpu")` (recommended for PTQ eval)

Then run the notebook cells in order.


## 5) Workflows

### 5.1 Baseline (FP32)
1. Train or fine‑tune ShuffleNet V2.
2. Save FP32 weights: `torch.save(model.state_dict(), "shufflenetv2_weights.pth")`.
3. Evaluate Top‑1 accuracy on `val` or `test` set.

### 5.2 Static PTQ (Per‑Tensor Symmetric)
- Prepare the model with `torch.ao.quantization` (observers + fusion if applicable).
- Run calibration on a few batches of **representative** `train`/`val` samples.
- Convert to quantized model and evaluate.

### 5.3 Per‑Channel Symmetric PTQ
- Same as Static PTQ, but configure **per‑channel** weight quantization for conv layers.

### 5.4 Power‑of‑Two (PoT) Quantization
- Constrain weight scales to powers of two.
- Evaluate impact on accuracy vs. compute simplicity.

### 5.5 AdaRound
- Perform learned rounding on weights with a small calibration/rounding set.
- Export and evaluate the rounded model.

> Tip: Keep the calibration subset fixed across methods to ensure fair comparison.


## 6) Measuring accuracy, size, and latency

- **Top‑1 accuracy**: The evaluation cell computes accuracy on `val` or `test`.
- **Model size**: Save the model and check file size.
  ```python
  torch.save(model.state_dict(), "model.pth")
  import os; print(os.path.getsize("model.pth")/1e6, "MB")
  ```
- **CPU latency**: Time the forward pass over several hundred images and average (warm‑up first).
  ```python
  import time
  model.eval()
  with torch.inference_mode():
      # warmup
      for _ in range(20): _ = model(example_input)
      t0 = time.time()
      iters = 200
      for _ in range(iters): _ = model(example_input)
      print((time.time()-t0)/iters*1000, "ms/img")
  ```

Make sure to use `torch.inference_mode()` and disable gradients during timing.


## 7) Reproducibility

- Set seeds where provided (`torch.manual_seed(42)`, `np.random.seed(42)`).
- Fix data transforms (resize/normalize) consistently across methods.
- Use the same evaluation split for all runs.


## 8) Converting notebooks to scripts (optional)

If you prefer CLI scripts, you can export with:

```bash
jupyter nbconvert --to script "Shufflenet V2 baseline.ipynb"
jupyter nbconvert --to script "Shufflenet static.ipynb"
jupyter nbconvert --to script "Shufflenet perchannel symmetric.ipynb"
jupyter nbconvert --to script "Shufflenet power of two.ipynb"
jupyter nbconvert --to script "Shufflenet adaround.ipynb"
```

Then edit paths at the top of the generated `.py` files and run with `python <script>.py`.


## 9) Troubleshooting

- **Torch / torchvision version mismatch**: Install a matching pair (see PyTorch install table).
- **No `fbgemm` backend** (non‑x86 CPU): INT8 speedups may be limited; use `qnnpack` or keep FP32.
- **Out‑of‑memory errors**: Lower batch size and ensure images are resized to 224×224.
- **Poor accuracy after PTQ**: Increase calibration set size, enable per‑channel for convs, or use AdaRound.


## 10) Citation / Acknowledgments

- ShuffleNet V2: Ningning Ma et al., *ECCV 2018*.
- PyTorch Quantization: `torch.ao.quantization` APIs.
- AdaRound: Bannerjee et al., *ICLR 2020* (learned rounding for PTQ).

---

**Maintainer note:** If your dataset path differs per notebook, search for `DATA_ROOT` / `data_dir` near the top and edit accordingly.
