import pandas as pd
from causal_graph_discovery import apply_causal_discovery

def generate_counterfactual(event_log, row, sensitive_attribute, causally_dependent_vars):
    counterfactual_row = row.copy()
    for causally_dependent_var in causally_dependent_vars:
        values = event_log[event_log[sensitive_attribute] != row[sensitive_attribute]][causally_dependent_var]
        counterfactual_amount = values.sample(n=1).values[0]
        counterfactual_row[causally_dependent_var] = counterfactual_amount
    counterfactual_row[sensitive_attribute] = (
        event_log[event_log[sensitive_attribute] != row[sensitive_attribute]][sensitive_attribute]
        .sample(n=1)
        .values[0]
    )
    return counterfactual_row

def generate_counterfactual_event_log(event_log, sensitive_attributes, labels, case_id_col):
    causal_graph_dependencies = apply_causal_discovery(event_log, labels, case_id_col, setting="fci", alpha=0.01)
    log_copy = event_log.copy()
    new_rows = []
    for _, group in log_copy.groupby(case_id_col):
        first_row = group.iloc[0]
        count = 0
        for sensitive_attribute in sensitive_attributes:
            causally_dependent_vars = []
            for tuple in causal_graph_dependencies:
                if tuple[0] == sensitive_attribute:
                    causally_dependent_vars.append(tuple[1])
            counterfactual = generate_counterfactual(event_log, first_row, sensitive_attribute, causally_dependent_vars)
            for col in causally_dependent_vars:
                log_copy.loc[group.index, col] = counterfactual[col]
            log_copy.loc[group.index, sensitive_attribute] = counterfactual[sensitive_attribute]
            log_copy.loc[group.index, "counterfactual_index"] = count
            count += 1
            new_rows.append(log_copy.loc[group.index])
    counterfactual_log = pd.concat(new_rows, ignore_index=True)
    counterfactual_log["counterfactual_index"] = counterfactual_log["counterfactual_index"].astype(int)
    return counterfactual_log

if __name__ == "__main__":
    data = pd.read_csv("data/hospital_log_medium.csv")
    labels = ['case:private_insurance', 'case:age', 'case:underlying_condition', 'case:citizen', 'case:gender']
    data[labels] = data[labels].astype(int)
    sensitive_attrs = ['case:gender', 'case:citizen']
    print("Generating counterfactual event log...")
    counterfactual_log = generate_counterfactual_event_log(data, sensitive_attrs, labels, "case:concept:name")
    # reconvert gender
    print(counterfactual_log.head())
    print("Done.")
    counterfactual_log.to_csv("data/hospital_log_medium_counterfactual.csv", index=False)