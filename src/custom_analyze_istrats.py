import torch
import numpy as np

# -----------------------------
# load interpretability output
# -----------------------------
d = torch.load(
    r"C:\Users\skyfi\projects\strats\outputs\aumc-4\istrats_final\aumc_istrats_interpret_infer.pt"
)
var_mean_contrib = d["var_mean_contrib"].cpu().numpy()

# -----------------------------
# variable list (ORDER MATTERS)
# -----------------------------
variables = [
    'plt','hct','crp','hgb','crea','k','o2sat','pco2','ph','bun','po2',
    'bicar','tnt','glu','be','alt','wbc','ast','ptt','inrpt','urine',
    'temp','sbp','dbp','ca','cl','hr','map','mg','phos','alb','resp',
    'dobu_rate','hbco','lact','dopa_rate','alp','bili','tri','bildir',
    'norepi_rate','epi_rate','pt','weight_static'
]

assert len(variables) == len(var_mean_contrib)

# -----------------------------
# rank and print
# -----------------------------
idx = np.argsort(-var_mean_contrib)

print("\nTop variables by mean I-STRATS contribution:\n")
for rank, i in enumerate(idx):
    print(
        f"{rank+1:2d}. {variables[i]:20s}  {var_mean_contrib[i]: .6f}"
    )
