import os
import torch

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")