# okc: problem=muse_bf16_proj
# SPDX-License-Identifier: Apache-2.0
# Original experimental implementation; no H200 benchmark claims.
import torch

def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    return state

def _torch(data, state):
    x, cfg = data["x"], data["config"]
    w = state[cfg["layer"]][:cfg["n"]]
    if cfg["layer"] == "router":
        if not x.is_cuda:
            return x.float() @ w.T
        hi = torch.mm(x, state["router_hi_t"][:, :cfg["n"]], out_dtype=torch.float32)
        lo = torch.mm(x, state["router_lo_t"][:, :cfg["n"]], out_dtype=torch.float32)
        return hi + lo
    return torch.nn.functional.linear(x, w)

def kernel_fn(data, state):
    return _torch(data, state)
