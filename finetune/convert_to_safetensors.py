import os
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

# Path setup
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

results_dir = project_root / "finetune" / "results_20260725_105616"
app_models_dir = project_root / "app" / "models"

app_models_dir.mkdir(parents=True, exist_ok=True)

from app.pipeline import SpectrumEncoder, BinaryClassifier


def convert_pt_to_safetensors():
    print("=" * 60)
    print("Converting 5-Fold official models from .pt to .safetensors format...")
    print("=" * 60)

    for fold in range(1, 6):
        pt_filename = f"official_ensemble_fold_{fold}.pt"
        st_filename = f"official_ensemble_fold_{fold}.safetensors"

        pt_path = results_dir / pt_filename
        if not pt_path.exists():
            print(f"[ERROR] Input file not found: {pt_path}")
            continue

        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

        # Build complete BinaryClassifier state_dict
        full_state_dict = {}
        if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt and "classifier_state_dict" in ckpt:
            for k, v in ckpt["encoder_state_dict"].items():
                full_state_dict[f"encoder.{k}"] = v.contiguous()
            for k, v in ckpt["classifier_state_dict"].items():
                full_state_dict[f"classifier.{k}"] = v.contiguous()
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            full_state_dict = {k: v.contiguous() for k, v in ckpt["state_dict"].items()}
        elif isinstance(ckpt, dict):
            full_state_dict = {k: v.contiguous() for k, v in ckpt.items()}

        # Verify model can be loaded successfully
        encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
        model = BinaryClassifier(encoder=encoder, input_dim=256, freeze_encoder=False)
        missing, unexpected = model.load_state_dict(full_state_dict, strict=True)
        print(f"Fold {fold}: State Dict validation passed! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

        # Save as .safetensors files to both results directory and app/models directory
        st_path_results = results_dir / st_filename
        st_path_app = app_models_dir / st_filename

        save_file(full_state_dict, str(st_path_results))
        save_file(full_state_dict, str(st_path_app))

        print(f"  [OK] Fold {fold} conversion successful:")
        print(f"       -> {st_path_results}")
        print(f"       -> {st_path_app}")

    print("\n[OK] All 5 Fold models successfully converted to .safetensors format!")


if __name__ == '__main__':
    convert_pt_to_safetensors()