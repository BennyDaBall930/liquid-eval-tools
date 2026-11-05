"""LFM3-1.7B NVFP4 helpers for the eval harness (kld.py, kld_exp.py, greedy_agree.py, dump_eval.py).

ModelOpt NVFP4 layout (checked on LiquidAI/LFM3-1.7B-NVFP4):
  <p>.weight          uint8 [rows, cols/2], two E2M1 codes per byte, LOW nibble = even column
  <p>.weight_scale    float8_e4m3fn [rows, cols/16] (linear layout, no swizzle)
  <p>.weight_scale_2  fp32 scalar gs; dequant w = e2m1(code) * e4m3(sf) * gs
  <p>.input_scale     fp32 scalar, static activation global scale (ModelOpt: calib amax / (6*448))
w1 and w3 share weight_scale_2 and input_scale (fused gate/up GEMM in vLLM/SGLang). Liquid's weights are plain RTN,
static-6 scales (gs = amax/2688). Non-FF tensors are bit-identical to the BF16 checkpoint.

Activation fake quant mirrors the W4A4 kernels: per 16-value block along K, sf = e4m3_rne(bmax/6 / input_scale)
(clamped to 448), x_q = e2m1_rne(x / (sf * input_scale)) * sf * input_scale (saturating at 6). The GEMM itself is
exact fp4 x fp4 products with fp32 accumulation, so fp32 F.linear on the dequantized operands reproduces it up to
summation order.
"""
import json, os
import torch
import torch.nn.functional as F

import nvfp4q as nq

# Local snapshots. Override with env vars; if the BF16 dir is missing, the pinned Hub revision is used instead.
BF16_DIR = os.environ.get("LFM3_BF16_DIR", r"C:\AI\Models\lfm3\bf16")              # LiquidAI/LFM3-1.7B
BF16_REV = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
LIQUID_DIR = os.environ.get("LFM3_LIQUID_DIR", r"C:\AI\Models\lfm3\liquid-nvfp4")  # LiquidAI/LFM3-1.7B-NVFP4


def bf16_source():
    """(path_or_repo_id, revision) for the BF16 teacher: the local snapshot if present, else the pinned Hub revision."""
    return (BF16_DIR, None) if os.path.isdir(BF16_DIR) else ("LiquidAI/LFM3-1.7B", BF16_REV)
N_LAYERS = 24
G = 16


def ff_names():
    return [f"model.layers.{i}.feed_forward.{w}" for i in range(N_LAYERS) for w in ("w1", "w3", "w2")]


def load_bf16(device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    src, rev = bf16_source()
    m = AutoModelForCausalLM.from_pretrained(src, revision=rev, trust_remote_code=True, dtype=dtype, device_map=device)
    m.eval()
    return m


def load_tokenizer():
    from transformers import AutoTokenizer
    src, rev = bf16_source()
    return AutoTokenizer.from_pretrained(src, revision=rev)


def act_quant(x, input_scale):
    """NVFP4 activation round trip along the last dim (multiple of 16). Returns float32."""
    shp = x.shape
    X = x.float().reshape(-1, shp[-1] // G, G)
    bmax = X.abs().amax(-1, keepdim=True)
    code, sf = nq.e4m3_rne((bmax / 6.0 / input_scale).clamp(max=448.0))
    inv = torch.where(sf > 0, 1.0 / (sf * input_scale), torch.zeros_like(sf))
    q = nq.e2m1_value(nq.e2m1_code(X * inv)) * sf * input_scale
    return q.reshape(shp)


def unpack_codes(packed):
    lo, hi = packed & 15, packed >> 4
    return torch.stack([lo, hi], -1).reshape(packed.shape[0], packed.shape[1] * 2)


def pack_codes(codes):
    c = codes.view(codes.shape[0], -1, 2)
    return (c[..., 0] | (c[..., 1] << 4)).contiguous()


def dequant(packed, sf, gs):
    codes = unpack_codes(packed)
    rows, cols = codes.shape
    v = nq.e2m1_value(codes).view(rows, cols // G, G) * sf.float().view(rows, cols // G, 1) * gs.float()
    return v.view(rows, cols)


def load_modelopt_ff(path, device="cuda"):
    """-> {prefix: dict(weight=W fp32 dequant, input_scale=float)} for every FF linear in a ModelOpt NVFP4 file."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, "pt", device=device) as f:
        for p in ff_names():
            W = dequant(f.get_tensor(p + ".weight"), f.get_tensor(p + ".weight_scale"), f.get_tensor(p + ".weight_scale_2"))
            out[p] = dict(weight=W, input_scale=float(f.get_tensor(p + ".input_scale").float()))
    return out


class FQLinear(torch.nn.Module):
    """Drop-in for an FF nn.Linear: W4A4 (input_scale set) or W4A16 (input_scale None) fake quant, fp32 GEMM."""
    def __init__(self, weight, input_scale):
        super().__init__()
        self.register_buffer("weight", weight.float())
        self.input_scale = input_scale

    def forward(self, x):
        xq = act_quant(x, self.input_scale) if self.input_scale is not None else x.float()
        return F.linear(xq, self.weight).to(x.dtype)


def patch_ff(model, qff, a4=True):
    """Swap FF linears for fake-quant ones; returns the originals so unpatch_ff can restore them."""
    orig = {}
    for p, d in qff.items():
        parent = model.get_submodule(p.rsplit(".", 1)[0])
        name = p.rsplit(".", 1)[1]
        orig[p] = getattr(parent, name)
        setattr(parent, name, FQLinear(d["weight"], d["input_scale"] if a4 else None))
    return orig


def unpatch_ff(model, orig):
    for p, mod in orig.items():
        parent = model.get_submodule(p.rsplit(".", 1)[0])
        setattr(parent, p.rsplit(".", 1)[1], mod)
