import math
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embedding_pretrain.lib.models import SpectrumEncoder

def evaluate():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 加载模型
    model_path = project_root / "embedding_pretrain" / "results_20260725_095428" / "best_model.pt"
    print(f"Loading checkpoint from: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu')

    epoch = ckpt.get('epoch', 'N/A')
    loss = ckpt.get('loss', None)
    if loss is None:
        loss = ckpt.get('best_loss', 0.0)

    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
    if 'encoder_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        # check if key starts with encoder.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_state_dict[k[len('encoder.'):]] = v
            elif k.startswith('module.encoder.'):
                new_state_dict[k[len('module.encoder.'):]] = v
            else:
                new_state_dict[k] = v
        encoder.load_state_dict(new_state_dict)
    else:
        encoder.load_state_dict(ckpt)

    encoder = encoder.to(device)
    encoder.eval()

    # 2. 加载数据
    data_path = project_root / "embedding_pretrain" / "data_source" / "preprocessed_spectra.parquet"
    if data_path.exists():
        table = pq.read_table(str(data_path), columns=['spectrum_vector'])
        raw_list = table['spectrum_vector'].to_pylist()
        spectra = np.array(raw_list, dtype=np.float32)
    else:
        npy_path = project_root / "embedding_pretrain" / "data_source" / "pretrain_spectra.npy"
        spectra = np.load(npy_path)

    num_eval_samples = min(10000, len(spectra))
    np.random.seed(42)
    indices = np.random.choice(len(spectra), num_eval_samples, replace=False)
    sampled_spectra = spectra[indices]
    spectra_tensor = torch.tensor(sampled_spectra, dtype=torch.float32).to(device)

    # 3. 提取特征
    with torch.no_grad():
        batch_size = 1000
        h_list = []
        for i in range(0, num_eval_samples, batch_size):
            batch_x = spectra_tensor[i:i+batch_size]
            h = encoder(batch_x)
            h_list.append(h)
        h_all = torch.cat(h_list, dim=0)  # [M, 256]

    # 4. 计算指标
    M = num_eval_samples
    d = h_all.shape[1]
    S = torch.mm(h_all, h_all.t())
    mask = ~torch.eye(M, dtype=torch.bool, device=device)
    sims = S[mask]

    avg_sim = sims.mean().item()
    std_sim = sims.std().item()

    sq_distances = 2 - 2 * S
    uniformity = torch.log(torch.exp(-2 * sq_distances)[mask].mean()).item()
    tolerance = torch.exp(-2 * sq_distances)[mask].mean().item()

    h_centered = h_all - h_all.mean(dim=0, keepdim=True)
    cov = torch.mm(h_centered.t(), h_centered) / (M - 1)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)
    p = eigenvalues / eigenvalues.sum()
    entropy = -torch.sum(p * torch.log(p)).item()
    effective_rank = math.exp(entropy)
    effective_ratio = (effective_rank - 1) / (d - 1)

    S_temp = S.clone()
    S_temp.fill_diagonal_(-float('inf'))
    nn_sim = torch.max(S_temp, dim=1)[0].mean().item()
    collapse_risk = avg_sim + nn_sim

    # 5. 格式化输出
    print(f"\n✓ 最佳模型 (loss: {loss:.6f})")
    print("  计算嵌入空间指标...")
    print(f"    Avg Similarity: {avg_sim:.4f} ± {std_sim:.4f}")
    print(f"    Uniformity: {uniformity:.4f} (越低越好)")
    print(f"    Tolerance: {tolerance:.4f} (越高越好)")
    print(f"    Effective Ratio: {effective_ratio:.3f}")
    print(f"    NN Similarity: {nn_sim:.4f}")
    print(f"    Collapse Risk: {collapse_risk:.2f}")

if __name__ == "__main__":
    evaluate()
