"""Import from every script: makes the `hsae` package (scripts/hsae) importable and chdir's to the
repo root, so the default data/ , models/ , results/ paths resolve no matter where a script is launched from."""
import os, sys
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
os.chdir(ROOT)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# cuBLAS on this box (torch 2.10+cu128, driver 550, H100) returns CUBLAS_STATUS_NOT_INITIALIZED for x[n,768] @ W[768,~100k]
# at a few small batch sizes (n = 6..8, exactly the beam widths used in the tests).  Re-express the failing product as
# (W @ x^T)^T, which uses a different cuBLAS path and works for every shape we probed.
import torch, torch.nn.functional as F
_orig_linear = F.linear
def _safe_linear(input, weight, bias=None):
    try:
        return _orig_linear(input, weight, bias)
    except RuntimeError as e:
        if "CUBLAS" not in str(e): raise
        x2 = input.reshape(-1, input.shape[-1]); y = (weight @ x2.t()).t()
        if bias is not None: y = y + bias
        return y.reshape(*input.shape[:-1], weight.shape[0])
F.linear = _safe_linear

if os.environ.get("HSAE_TF32") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
