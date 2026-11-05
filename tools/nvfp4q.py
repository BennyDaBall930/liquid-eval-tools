"""NVFP4 grid primitives used by the eval harness (lfm3q.py): E2M1 code <-> value and E4M3 round-to-nearest-even.
Dequant everywhere is w = (E2M1(code) * e4m3(sf)) * per-tensor scale, the same formula vLLM, SGLang, llama.cpp and
ModelOpt use for NVFP4.
"""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
G = 16  # NVFP4 sub-block (one E4M3 scale per 16 weights along the input dim)


def e4m3_rne(v):
    """float32 >= 0 -> (code uint8, decoded float32). RNE with subnormals; caller guarantees v <= 448."""
    f8 = v.to(torch.float8_e4m3fn)
    return f8.view(torch.uint8), f8.to(torch.float32)


def e2m1_code(y):
    """float32 -> E2M1 code (bit 3 = sign), RNE on the grid {0,.5,1,1.5,2,3,4,6}, saturating at 6."""
    a = y.abs()
    mag = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    return mag | (((y < 0) & (mag > 0)).to(torch.uint8) << 3)


def e2m1_value(code):
    v = E2M1.to(code.device)[(code & 7).long()]
    return torch.where((code & 8) > 0, -v, v)


def dequant_codes(codes, scodes, gs):
    """Reference dequant from unpacked codes: (E2M1 * e4m3) * gs, float32 [rows, cols]."""
    rows, cols = codes.shape
    sf = scodes.view(torch.float8_e4m3fn).to(torch.float32).reshape(rows, cols // G, 1)
    v = e2m1_value(codes).reshape(rows, cols // G, G)
    return ((v * sf) * gs).reshape(rows, cols)
