import json
import os

nb_path = os.path.join("simclr_finetune", "预训练+二分类.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/nb_content.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            out.write(f"==================== Cell {i} ====================\n")
            out.write(source + "\n\n")
