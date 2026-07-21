import torch
import os
import json

f1 = os.path.join("simclr_finetune", "binary_classifier_latest.pt")
ckpt1 = torch.load(f1, map_location="cpu", weights_only=False)

for k in ['train_results', 'val_results', 'test_results']:
    if k in ckpt1:
        print(f"\n--- Stored {k} in binary_classifier_latest.pt ---")
        res = ckpt1[k]
        for sub_k, val in res.items():
            if not isinstance(val, (list, dict, torch.Tensor)):
                print(f"  {sub_k}: {val}")
            elif hasattr(val, 'shape'):
                print(f"  {sub_k} shape: {val.shape}")
