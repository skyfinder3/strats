import torch
import numpy as np


d = torch.load("..._istrats_interpret_infer.pt")
var_mean_contrib = d["var_mean_contrib"].numpy()  # shape (V,)

# e.g., print top-10 variables by contribution
idx = np.argsort(-var_mean_contrib)
for rank, v in enumerate(idx[:10]):
    print(rank+1, v, var_mean_contrib[v])