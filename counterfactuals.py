import pandas as pd

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

def generate_counterfactual_event_log(event_log, sensitive_attribute, causally_dependent_vars, case_id_col):
    counterfactual_log = event_log.copy()
    for case_id, group in counterfactual_log.groupby(case_id_col):
        first_row = group.iloc[0]
        counterfactual = generate_counterfactual(event_log, first_row, sensitive_attribute, causally_dependent_vars)
        for col in causally_dependent_vars:
            counterfactual_log.loc[group.index, col] = counterfactual[col]
        counterfactual_log.loc[group.index, sensitive_attribute] = counterfactual[sensitive_attribute]
    return counterfactual_log

# if __name__ == "__main__":
#     data = pd.read_pickle("data/bpi12.pkl")
#     data.to_csv("data/bpi12_kappel.csv", index=False)
#     sensitive_attr = "gender"
#     dependent_vars = ["case:AMOUNT_REQ"]
#     sample_row = data.sample(n=1).iloc[0]
#     print("Generating counterfactual event log...")
#     counterfactual_log = generate_counterfactual_event_log(data, sensitive_attr, dependent_vars, "case_id")
#     print(counterfactual_log.head())
#     print("Done.")
#     counterfactual_log.to_csv("data/bpi12_counterfactual.csv", index=False)