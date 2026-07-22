import os
import torch
from safetensors.torch import save_file, load_file

def convert_pt_to_safetensors(pt_path: str, safetensors_path: str):
    print(f"Loading PyTorch checkpoint from: {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    combined_state_dict = {}
    
    if "encoder_state_dict" in ckpt:
        for k, v in ckpt["encoder_state_dict"].items():
            combined_state_dict[f"encoder.{k}"] = v.contiguous()
    else:
        print("Warning: 'encoder_state_dict' not found in checkpoint directly.")
        
    if "classifier_state_dict" in ckpt:
        for k, v in ckpt["classifier_state_dict"].items():
            combined_state_dict[f"classifier.{k}"] = v.contiguous()
    else:
        print("Warning: 'classifier_state_dict' not found in checkpoint directly.")

    # If key format is directly state_dict
    if not combined_state_dict and isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor):
                combined_state_dict[k] = v.contiguous()

    os.makedirs(os.path.dirname(safetensors_path), exist_ok=True)
    save_file(combined_state_dict, safetensors_path)
    print(f"Successfully converted and saved safetensors model to: {safetensors_path}")

    # Verify loading
    loaded_tensors = load_file(safetensors_path)
    print(f"Verification successful: Loaded {len(loaded_tensors)} tensors from safetensors.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pt_path = os.path.join(project_root, "simclr_finetune", "binary_classifier_weights_20260716_164027.pt")
    safetensors_path = os.path.join(project_root, "app", "models", "binary_classifier_weights_20260716_164027.safetensors")
    
    convert_pt_to_safetensors(pt_path, safetensors_path)
