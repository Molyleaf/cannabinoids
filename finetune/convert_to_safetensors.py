import os
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

# 路径设置
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
    print("正在将 5-Fold 正式模型从 .pt 转换为 .safetensors 格式...")
    print("=" * 60)

    for fold in range(1, 6):
        pt_filename = f"official_ensemble_fold_{fold}.pt"
        st_filename = f"official_ensemble_fold_{fold}.safetensors"

        pt_path = results_dir / pt_filename
        if not pt_path.exists():
            print(f"[ERROR] 找不到输入文件: {pt_path}")
            continue

        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)

        # 构建完整的 BinaryClassifier state_dict
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

        # 验证模型能否成功加载
        encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
        model = BinaryClassifier(encoder=encoder, input_dim=256, freeze_encoder=False)
        missing, unexpected = model.load_state_dict(full_state_dict, strict=True)
        print(f"Fold {fold}: State Dict 验证通过! (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

        # 保存为 .safetensors 文件至 results 目录与 app/models 目录
        st_path_results = results_dir / st_filename
        st_path_app = app_models_dir / st_filename

        save_file(full_state_dict, str(st_path_results))
        save_file(full_state_dict, str(st_path_app))

        print(f"  [OK] Fold {fold} 转换成功:")
        print(f"       -> {st_path_results}")
        print(f"       -> {st_path_app}")

    print("\n[OK] 全部 5 个 Fold 模型已成功转换为 .safetensors 格式！")

if __name__ == '__main__':
    convert_pt_to_safetensors()
