import json
import os

nb_path = os.path.join("simclr_finetune", "预训练+二分类.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "class " in source or "def " in source or "torch.load" in source or "binary" in source.lower():
            print(f"--- Cell {i} ---")
            print(source[:1000])
            print("\n")
