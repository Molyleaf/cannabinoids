import torch
import os

finetune_dir = os.path.join("simclr_finetune")
files = [f for f in os.listdir(finetune_dir) if f.endswith(".pt")]
print("Found .pt files in simclr_finetune:", files)

for fname in files:
    fpath = os.path.join(finetune_dir, fname)
    print(f"\n--- Inspecting {fname} ---")
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        continue
        
    if isinstance(data, dict):
        print("Type: dict")
        print("Keys:", list(data.keys()))
        for k in data.keys():
            if isinstance(data[k], dict):
                print(f"  sub-dict {k} keys (sample):", list(data[k].keys())[:5])
            elif isinstance(data[k], torch.Tensor):
                print(f"  tensor {k} shape:", data[k].shape)
            else:
                print(f"  key {k}:", type(data[k]), data[k] if not isinstance(data[k], (list, dict)) else f"len={len(data[k])}")
    else:
        print("Type:", type(data))
