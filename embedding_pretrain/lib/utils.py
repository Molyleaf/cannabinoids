import matplotlib.pyplot as plt
import numpy as np


class ConvergenceChecker:
    """检测模型是否拟合"""
    def __init__(self, patience=20, min_delta=1e-4, window_size=10):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.losses = []
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        
    def update(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.epochs_no_improve = 0
            return False
        else:
            self.epochs_no_improve += 1
            
            # 额外检查：最近window_size个epoch的损失标准差是否很小
            if len(self.losses) >= self.window_size:
                recent = self.losses[-self.window_size:]
                if np.std(recent) < self.min_delta * 5:
                    return True  # 损失不再变化，已拟合
            
            return self.epochs_no_improve >= self.patience
    
    def is_converged(self):
        return self.epochs_no_improve >= self.patience

def plot_loss_history(loss_history, best_loss, output_path):
    """绘制并保存损失曲线 (同时保存 CSV 文件与 PNG 图片)"""
    import pandas as pd
    from pathlib import Path

    out_p = Path(output_path)
    csv_path = out_p.with_suffix('.csv')
    
    df = pd.DataFrame({
        'epoch': list(range(1, len(loss_history) + 1)),
        'loss': loss_history
    })
    df.to_csv(csv_path, index=False)
    print(f"[OK] 已导出预训练 Loss 数据至 CSV: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', alpha=0.7)
    plt.axhline(y=best_loss, color='r', linestyle='--', alpha=0.5,
                label=f'Best: {best_loss:.4f}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('NT-Xent Loss', fontsize=12)
    plt.title('SimCLR Pre-training Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_p), dpi=150)
    plt.close()

