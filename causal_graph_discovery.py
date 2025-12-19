from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd

def apply_causal_discovery(df, labels, case_id_column, setting="pc", alpha=0.01):
    df = df.groupby(case_id_column).first().reset_index()
    df = df[labels]
    df = df.to_numpy()
    if setting == "pc":
        cg = pc(df, alpha=alpha, stable=True)
        pyd = GraphUtils.to_pydot(cg.G, labels=labels)
        pyd.write_png('simple_test_pc.png')
    elif setting == "fci":
        cg, _ = fci(df)
        pyd = GraphUtils.to_pydot(cg.G, labels=labels)
        pyd.write_png('simple_test_fci.png')
    else:
        raise ValueError("Unsupported setting. Use 'pc' or 'fci'.")