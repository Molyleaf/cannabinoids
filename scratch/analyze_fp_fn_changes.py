import os
import sys

import torch

PROJECT_ROOT = r"d:\UserFiles\Documents\PyCharm\cannabinoids"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simclr_finetune.lib.dataset import prepare_finetune_dataset
from simclr_finetune.lib.models import SpectrumEncoder, BinaryClassifier
from simclr_finetune.lib.evaluation import evaluate_and_record_predictions

def main():
    pos_msp = os.path.join(PROJECT_ROOT, "simclr_finetune", "data_source", "阳性-含CanonicalSMILES-5类骨架(4).msp")
    if not os.path.exists(pos_msp):
        pos_msp = os.path.join(PROJECT_ROOT, "simclr_finetune", "data_source", "阳性-含CanonicalSMILES-5类骨架(3).msp")
    neg_msp = os.path.join(PROJECT_ROOT, "simclr_finetune", "data_source", "阴性(4).msp")

    # Load dataset
    train_loader, train_eval_loader, val_loader, test_loader, meta = prepare_finetune_dataset(
        pos_msp_path=pos_msp,
        neg_msp_path=neg_msp,
        batch_size=128,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    # 1. Load newly trained fully-aligned clean_spectrum model
    model_clean_path = os.path.join(PROJECT_ROOT, "simclr_finetune", "results_20260723_204354_fixed", "binary_classifier_20260723_204354.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SpectrumEncoder(input_dim=561, hidden_dim=256)
    model = BinaryClassifier(encoder=encoder, input_dim=256, freeze_encoder=True)
    
    ckpt = torch.load(model_clean_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model = ckpt
    model.to(device)
    model.eval()

    print("=== Analyzing FP and FN Confusion Matrix across Train / Val / Test Sets ===")

    # Evaluate on Train, Val, Test
    train_res = evaluate_and_record_predictions(model, train_eval_loader, sample_names=meta['train_sample_names'], device=device)
    val_res = evaluate_and_record_predictions(model, val_loader, sample_names=meta['val_sample_names'], device=device)
    test_res = evaluate_and_record_predictions(model, test_loader, sample_names=meta['test_sample_names'], device=device)

    sets_data = [('Train', train_res), ('Val', val_res), ('Test', test_res)]

    for set_name, res in sets_data:
        cm = res['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        acc = res['accuracy']
        prec = res['precision']
        rec = res['recall']
        f1 = res['f1']
        auc = res['auc']

        print(f"\n--- {set_name} Set (Total: {res['n_samples']}) ---")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

    # Inspect specific FP and FN compounds in Train set
    print("\n" + "="*80)
    print("Top FP (False Positive) Compounds in Train Set (Negatives misclassified as Positives):")
    train_preds = train_res['df']
    fps = train_preds[(train_preds['true_label'] == 0) & (train_preds['pred_label'] == 1)].sort_values('prob', ascending=False)
    for idx, row in fps.head(10).iterrows():
        print(f"  Name: {row['sample_name'][:40]:<40} | Prob: {row['prob']*100:6.2f}%")

    print("\nTop FN (False Negative) Compounds in Train Set (Positives misclassified as Negatives):")
    fns = train_preds[(train_preds['true_label'] == 1) & (train_preds['pred_label'] == 0)].sort_values('prob', ascending=True)
    for idx, row in fns.head(10).iterrows():
        print(f"  Name: {row['sample_name'][:40]:<40} | Prob: {row['prob']*100:6.2f}%")

if __name__ == "__main__":
    main()
