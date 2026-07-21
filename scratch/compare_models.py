import torch
import os

f1 = os.path.join("simclr_finetune", "binary_classifier_latest.pt")
f2 = os.path.join("simclr_finetune", "binary_classifier_weights_20260716_164027.pt")

ckpt1 = torch.load(f1, map_location="cpu", weights_only=False)
ckpt2 = torch.load(f2, map_location="cpu", weights_only=False)

print("Checking binary_classifier_latest.pt vs binary_classifier_weights_20260716_164027.pt...")

enc_same = True
for k, v in ckpt1["encoder_state_dict"].items():
    if not torch.equal(v, ckpt2["encoder_state_dict"][k]):
        print(f"Encoder key {k} differs!")
        enc_same = False

cls_same = True
for k, v in ckpt1["classifier_state_dict"].items():
    if not torch.equal(v, ckpt2["classifier_state_dict"][k]):
        print(f"Classifier key {k} differs!")
        cls_same = False

if enc_same and cls_same:
    print("Both checkpoint files have IDENTICAL weights!")
else:
    print(f"Encoder weights same: {enc_same}, Classifier weights same: {cls_same}")

if "timestamp" in ckpt1:
    print("Latest model saved timestamp:", ckpt1.get("timestamp"))
