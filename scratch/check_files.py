import os
import torch
import numpy as np

neg_path1 = r"d:\UserFiles\Documents\PyCharm\SimCLR\scratch\阴性.msp"
neg_path2 = r"d:\UserFiles\Documents\PyCharm\cannabinoids\scratch\阴性.msp"

pos_path1 = r"d:\UserFiles\Documents\PyCharm\SimCLR\known_compound_recognize\positive.msp"
pos_path2 = r"d:\UserFiles\Documents\PyCharm\cannabinoids\known_compound_recognize\positive.msp"

print("neg_path1 exists:", os.path.exists(neg_path1))
print("neg_path2 exists:", os.path.exists(neg_path2))

print("pos_path1 exists:", os.path.exists(pos_path1))
print("pos_path2 exists:", os.path.exists(pos_path2))
