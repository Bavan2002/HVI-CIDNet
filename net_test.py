# ===== Model Complexity Benchmark =====
# Measures parameters, FLOPs, and inference time for CIDNet
# Usage: python net_test.py

from thop import profile  # PyTorch-OpCounter for FLOPs calculation
import torch
import time
from net.CIDNet import CIDNet

# Initialize model on GPU
model = CIDNet().to("cuda")
input = torch.rand(1, 3, 256, 256).to("cuda")  # Standard benchmark size: 256x256 RGB

# ===== Measure Inference Time =====
torch.cuda.synchronize()  # Ensure GPU ops are complete before timing
model.eval()  # Set to evaluation mode (disables dropout, etc.)

time_start = time.time()
_ = model(input)  # Single forward pass
time_end = time.time()

torch.cuda.synchronize()  # Wait for GPU to finish
time_sum = time_end - time_start
print(f"Time: {time_sum}")

# ===== Count Parameters =====
n_param = sum([p.nelement() for p in model.parameters()])  # Total trainable params
n_paras = f"n_paras: {(n_param / 2**20)}M\n"  # Convert to millions (2^20)
print(n_paras)

# ===== Count FLOPs =====
macs, params = profile(model, inputs=(input,))  # MACs = Multiply-Accumulate ops
print(f"FLOPs:{macs / (2**30)}G")  # Convert to GFLOPs (2^30)
