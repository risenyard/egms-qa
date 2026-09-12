# Python and CUDA environment

Start with the [installation commands](../README.md#installation) to clone the
repository and create `.venv`. All commands here run from that checkout with
`.venv` activated. A Python environment contains the packages; NVIDIA drivers
and GPU access are provided by the machine or cluster.

## Choose the environment

Use Python 3.10 on Linux for the verified full-workflow setup. Python 3.12 was
also checked for core, `tasks`, and `translator` installation; full workflows
were exercised on Python 3.10. To use 3.12, replace `python3.10` with
`python3.12` in the environment-creation command. Other operating systems and
Python versions have not received the same workflow validation.

| Workload | Package installation | Compute |
|---|---|---|
| Encoder, label aggregation, QA generation | `python -m pip install -e .` | CPU supported; GPU recommended for encoder training and bulk extraction |
| Task reference-value computation | `python -m pip install -e '.[tasks]'` | CPU for most groups; CUDA for the encoder-dependent A1/A2 groups |
| Translator question answering, training, evaluation | `python -m pip install -e '.[translator]'` | NVIDIA GPU with CUDA and bfloat16 support |
| All workflows | `python -m pip install -e '.[tasks,translator]'` | CPU and CUDA as above |

Extras add dependencies to the same environment; do not create a separate
checkout or environment for each component. Reading QA directly with
`load_dataset()` additionally requires `python -m pip install datasets`, as
shown in the [Dataset card](https://huggingface.co/datasets/risenyard/egms-qa-dataset#qa-use).

For a CPU-only machine, select a CPU PyTorch build before installing the
project to avoid downloading CUDA packages. For GPU use, select a PyTorch
CUDA build compatible with your NVIDIA driver. Use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/),
run its command inside `.venv`, then run the appropriate project installation
command above. EGMS-QA does not require `torchvision` or `torchaudio`.

## Verified package versions

The following combination was used for complete workflows on Linux x86-64.
These are reference versions, not a lockfile or a requirement to replace a
working GPU driver. Supported dependency ranges remain in
[`pyproject.toml`](../pyproject.toml); normal installation may resolve newer
versions, so it does not promise this exact combination or identical numerical
results.

| Component | Verified version |
|---|---|
| Python | 3.10.21 |
| PyTorch / packaged CUDA runtime | 2.14.0 / 13.0 (`torch.__version__`: `2.14.0+cu130`) |
| NumPy / pandas / PyArrow | 2.2.6 / 2.3.3 / 25.0.1 |
| safetensors / huggingface_hub | 0.8.0 / 1.30.0 |
| SciPy / scikit-learn | 1.15.3 / 1.7.2 |
| joblib / Matplotlib | 1.6.0 / 3.10.9 |
| Transformers / PEFT | 5.16.1 / 0.20.0 |
| Accelerate / bitsandbytes | 1.14.0 / 0.50.2 |
| quantulum3 / datasets | 0.10.0 / 5.0.1 |

GPU workflows were checked on NVIDIA L40, L40S, and RTX PRO 6000 Blackwell
Server Edition GPUs with driver 595.58.03. This is a tested setup, not a minimum
hardware specification. GPU memory requirements depend on the host model,
batch size, and sequence length; loading a model successfully does not by
itself establish that its training batch will fit.

## Check the environment

Check the interpreter, installed dependencies, and a real CPU tensor operation:

```bash
python --version
python -m pip --version
python -m pip check
python - <<'PY'
import sys
import torch
import egms_encoder
import egms_qa
print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Packaged CUDA:", torch.version.cuda)
print("CPU tensor:", (torch.ones(2) + 1).tolist())
print("CUDA available:", torch.cuda.is_available())
PY
```

The Python and pip paths should belong to `.venv`, and `pip check` should report
no broken requirements. `CUDA available: False` is expected on a CPU-only
machine. For GPU workflows, run the following on the actual GPU machine or
inside a GPU allocation, not on a cluster login node:

```bash
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: check GPU allocation, driver, and PyTorch build."
assert torch.cuda.is_bf16_supported(), "The released translators require bfloat16 support."
x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
y = x @ x
assert torch.isfinite(y).all().item()
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA / bfloat16 operation: OK")
PY
```

This verifies GPU access and an actual bfloat16 operation. Continue with the
[one-question example](../src/egms_qa/translator/README.md#ask-one-question)
to check a released model end to end.

## Common setup problems

| Symptom | Action |
|---|---|
| `python3.10: command not found` or `venv` / `ensurepip` unavailable | Install Python 3.10 with venv support using your OS package manager, or use an available Python 3.12 interpreter. Then repeat environment creation. |
| `ModuleNotFoundError` or a missing `egms-qa-*` command | Activate `.venv`, return to the checkout, and install the required extra with `python -m pip`. Use `python -m pip --version` to check which environment is receiving packages. |
| `nvidia-smi` fails | Check the NVIDIA driver and whether a GPU is exposed to your session or container; on a cluster, obtain a GPU allocation first. |
| `nvidia-smi` works but PyTorch reports CUDA unavailable | Check `torch.version.cuda` and the PyTorch build/driver combination using the installation selector above. |
| CUDA out of memory | Free GPU memory or reduce the training batch size; the host model must still fit. |
| Hugging Face downloads time out | Check network access from the machine running the command. For offline execution, download public artifacts on a connected machine and supply the documented local paths. |

To leave the environment, run `deactivate`. To return in another terminal,
change to the checkout and run `source .venv/bin/activate` again.
