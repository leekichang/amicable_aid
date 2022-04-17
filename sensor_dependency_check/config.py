import os
import torch
import warnings
import sys
warnings.filterwarnings(action='ignore')

USE_CUDA        = torch.cuda.is_available()
DEVICE          = torch.device("cuda" if USE_CUDA else "cpu")

print(f"WORKING WITH {DEVICE}", file = sys.stderr)

