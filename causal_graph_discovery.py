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
        nodes = cg.get_nodes()
        edges = cg.get_graph_edges()
        print("Nodes:")
        for node in nodes:
            print(f"{node.get_name()}")
        print("Edges:")
        edge_list = []
        for edge in edges:
            source_idx = int(edge.get_node1().get_name().replace("X", "")) - 1
            destination_idx = int(edge.get_node2().get_name().replace("X", "")) - 1
            source = labels[source_idx]
            destination = labels[destination_idx]
            edge_list.append((source, destination))
            print(f"{source} -> {destination}")
        pyd = GraphUtils.to_pydot(cg, labels=labels)
        pyd.write_png('simple_test_fci.png')
        return edge_list
    else:
        raise ValueError("Unsupported setting. Use 'pc' or 'fci'.")
    
if __name__ == "__main__":
    df = pd.read_csv('data/hospital_log_medium.csv')
    # mantain only the first row for each case_id
    df = df.groupby('case:concept:name').first().reset_index()
    print(df.columns)
    labels = ['case:private_insurance', 'case:age', 'case:underlying_condition', 'case:citizen', 'case:gender']
    # convert values from boolean to int
    df[labels] = df[labels].astype(int)
    case_id_column = 'case:concept:name'
    apply_causal_discovery(df, labels, case_id_column, setting="fci", alpha=0.01)