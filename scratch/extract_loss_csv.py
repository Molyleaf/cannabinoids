import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

def extract_loss_csv():
    img_path = Path("embedding_pretrain/results_20260725_095428/pretraining_loss.png")
    if not img_path.exists():
        print(f"Error: {img_path} not found")
        return

    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # Blue curve mask: dominant blue color
    blue_mask = (b.astype(int) - r.astype(int) > 40) & (b.astype(int) - g.astype(int) > 40)

    # 500 Epochs mapping:
    # Tick 0 (0 epochs): x = 145.5
    # Tick 5 (500 epochs): x = 1413.5
    # Step = (1413.5 - 145.5) / 500 = 2.536 pixels / epoch
    records = []
    for epoch in range(1, 501):
        x_center = 145.5 + 2.536 * epoch
        x_low = int(np.floor(x_center - 0.5))
        x_high = int(np.ceil(x_center + 0.5))
        
        ys = []
        for x in range(max(86, x_low), min(1478, x_high + 1)):
            y_pts = np.where(blue_mask[:, x])[0]
            ys.extend(y_pts)
            
        if len(ys) > 0:
            y_avg = np.mean(ys)
        else:
            y_avg = np.nan
            
        records.append({'epoch': int(epoch), 'y_pixel': y_avg})

    df = pd.DataFrame(records)
    df['y_pixel'] = df['y_pixel'].interpolate(method='linear').bfill().ffill()

    # Y Axis mapping:
    # Y Tick at y=732.5 corresponds to Loss = 0.60
    # 960 pixels correspond to 1.0 loss unit (96 pixels per 0.1 loss)
    # Loss(y) = 0.60 - (y_pixel - 732.5) / 960.0
    df['loss'] = 0.60 - (df['y_pixel'] - 732.5) / 960.0

    # Clean DataFrame to keep epoch and loss (rounded to 6 decimal places)
    df_clean = pd.DataFrame({
        'epoch': df['epoch'].astype(int),
        'loss': df['loss'].round(6)
    })

    # Save to CSV in results directory and top-level embedding_pretrain directory
    out_dir = img_path.parent
    csv_results_path = out_dir / "pretraining_loss.csv"
    csv_main_path = img_path.parent.parent / "pretraining_loss.csv"

    df_clean.to_csv(csv_results_path, index=False)
    df_clean.to_csv(csv_main_path, index=False)

    print(f"[OK] Extracted loss curve saved to:\n  1. {csv_results_path}\n  2. {csv_main_path}")
    print("\nLoss statistics:")
    print(df_clean.describe())

    print("\nFirst 10 Epochs:")
    print(df_clean.head(10).to_string(index=False))

    print("\nLast 10 Epochs:")
    print(df_clean.tail(10).to_string(index=False))

    min_row = df_clean.loc[df_clean['loss'].idxmin()]
    print(f"\nMinimum loss: {min_row['loss']} at Epoch {int(min_row['epoch'])}")

if __name__ == "__main__":
    extract_loss_csv()
