import pandas as pd
from causal_graph_discovery import apply_causal_discovery
import networkx as nx
from dowhy import gcm
from scipy.stats import norm

def generate_counterfactual(event_log, row, sensitive_attribute, causally_dependent_vars):
    counterfactual_row = row.copy()
    for causally_dependent_var in causally_dependent_vars:
        values = event_log[event_log[sensitive_attribute] != row[sensitive_attribute]][causally_dependent_var]
        counterfactual_amount = values.sample(n=1).values[0]
        counterfactual_row[causally_dependent_var] = counterfactual_amount
    if sensitive_attribute == "age":
        value = row[sensitive_attribute]
        counterfactual_row[sensitive_attribute] = 65 + (65 - value) if value < 65 else 65 - (value - 65)
    else:
        counterfactual_row[sensitive_attribute] = (
            event_log[event_log[sensitive_attribute] != row[sensitive_attribute]][sensitive_attribute]
            .sample(n=1)
            .values[0]
        )
    return counterfactual_row

import numpy as np
import pandas as pd
import networkx as nx

from dowhy import gcm

def generate_counterfactuals_dowhy_bpi13(
    event_log,
    sensitive_attrs,
    labels,
    case_id_col,
    categorical_target="product",
    random_state=0,
):
    causal_graph_dependencies = apply_causal_discovery(
        event_log,
        labels,
        case_id_col,
        setting="fci",
        alpha=0.01,
    )

    rng = np.random.default_rng(random_state)

    event_log_case_level = event_log.drop_duplicates(
        subset=[case_id_col],
        keep="first",
    ).copy()

    event_log_model = event_log_case_level[labels].copy()

    event_log_model[categorical_target] = (
        event_log_model[categorical_target]
        .astype("string")
    )

    dependencies = list(causal_graph_dependencies)
    

    event_logs_counterfactuals = []  # To store counterfactual logs for each sensitive attribute
    count = 0
    for sensitive_attr in sensitive_attrs:

        causal_graph = nx.DiGraph(dependencies)
        causal_graph.remove_node("organization involved")
        print(causal_graph)
        event_log_counterfactual = event_log.copy()

        causal_model = gcm.StructuralCausalModel(causal_graph)

        # Root sensitive attribute: empirical marginal distribution.
        causal_model.set_causal_mechanism(
            sensitive_attr,
            gcm.EmpiricalDistribution(),
        )

        causal_model.set_causal_mechanism(
            "impact",
            gcm.EmpiricalDistribution(),
        )

        causal_model.set_causal_mechanism(
            categorical_target,
            gcm.ClassifierFCM(
                classifier_model=gcm.ml.create_logistic_regression_classifier()
            ),
        )

        gcm.fit(causal_model, event_log_model)

        # Build intervention that picks a different value for the sensitive attribute
        # from the set of observed values (handles non-binary attributes).
        possible_values = list(event_log_model[sensitive_attr].unique())
        def _swap_value(g, vals=possible_values):
            # choose a value different from g if possible, otherwise return g
            alternatives = [v for v in vals if v != g]
            return rng.choice(alternatives) if alternatives else g

        intervention = {sensitive_attr: _swap_value}

        counterfactual_sensitive_values = []
        counterfactual_case_types = []

        for idx in range(len(event_log_model)):
            df_row = event_log_model.iloc[[idx]].copy()

            noise_row = pd.DataFrame(
                {
                    sensitive_attr: [rng.choice(event_log_model[sensitive_attr].values)],
                    "impact": [rng.choice(event_log_model["impact"].values)],
                    categorical_target: [rng.uniform(0.0, 1.0)],
                },
                index=df_row.index,
            )

            counterfactual_df = gcm.counterfactual_samples(
                causal_model,
                interventions=intervention,
                noise_data=noise_row,
            )

            counterfactual_sensitive_values.append(
                counterfactual_df[sensitive_attr].values[0]
            )
            counterfactual_case_types.append(
                counterfactual_df[categorical_target].values[0]
            )

        df_counterfactual = event_log_case_level.copy()
        df_counterfactual[sensitive_attr] = counterfactual_sensitive_values
        df_counterfactual[categorical_target] = counterfactual_case_types

        counterfactual_map_sensitive = dict(
            zip(event_log_case_level[case_id_col], counterfactual_sensitive_values)
        )
        counterfactual_map_case_type = dict(
            zip(event_log_case_level[case_id_col], counterfactual_case_types)
        )

        event_log_counterfactual[sensitive_attr] = (
            event_log_counterfactual[case_id_col].map(counterfactual_map_sensitive)
        )
        event_log_counterfactual[categorical_target] = (
            event_log_counterfactual[case_id_col].map(counterfactual_map_case_type)
        )
        event_log_counterfactual["counterfactual_index"] = count
        event_logs_counterfactuals.append(event_log_counterfactual)
        count += 1

    event_log_counterfactual = pd.concat(event_logs_counterfactuals, ignore_index=True)

    return event_log_counterfactual

def generate_counterfactuals_dowhy_hospital_billing(
    event_log,
    sensitive_attrs,
    labels,
    case_id_col,
    categorical_target="caseType",
    random_state=0,
):
    causal_graph_dependencies = apply_causal_discovery(
        event_log,
        labels,
        case_id_col,
        setting="fci",
        alpha=0.01,
    )

    rng = np.random.default_rng(random_state)

    event_log_case_level = event_log.drop_duplicates(
        subset=[case_id_col],
        keep="first",
    ).copy()

    event_log_model = event_log_case_level[labels].copy()

    event_log_model[categorical_target] = (
        event_log_model[categorical_target]
        .astype("string")
    )

    dependencies = list(causal_graph_dependencies)
    

    event_logs_counterfactuals = []  # To store counterfactual logs for each sensitive attribute
    count = 0
    for sensitive_attr in sensitive_attrs:

        causal_graph = nx.DiGraph(dependencies)
        event_log_counterfactual = event_log.copy()
        print(sensitive_attr)
        if sensitive_attr == "age":
            # remove "gender" as a parent of "age" if it exists
            if ("gender", "age") in causal_graph.edges:
                causal_graph.remove_node("gender")
            if ("gender", "caseType") in causal_graph.edges:
                causal_graph.remove_node("gender")
        elif sensitive_attr == "gender":
            # remove "age" as a parent of "gender" if it exists
            if ("age", "gender") in causal_graph.edges:
                causal_graph.remove_node("age")
            if ("age", "caseType") in causal_graph.edges:
                causal_graph.remove_node("age")
        

        causal_model = gcm.StructuralCausalModel(causal_graph)

        # Root sensitive attribute: empirical marginal distribution.
        causal_model.set_causal_mechanism(
            sensitive_attr,
            gcm.EmpiricalDistribution(),
        )

        causal_model.set_causal_mechanism(
            categorical_target,
            gcm.ClassifierFCM(
                classifier_model=gcm.ml.create_logistic_regression_classifier()
            ),
        )

        if ("gender", "age") in causal_graph.edges:
            causal_model.set_causal_mechanism(
                "age",
                gcm.AdditiveNoiseModel(
                    prediction_model=gcm.ml.create_linear_regressor(),
                    noise_model=gcm.ScipyDistribution(norm),
                ),
            )

        gcm.fit(causal_model, event_log_model)

        intervention = {
            sensitive_attr: (lambda g: 2 if g != 2 else 1)
        }

        counterfactual_sensitive_values = []
        counterfactual_case_types = []

        for idx in range(len(event_log_model)):
            df_row = event_log_model.iloc[[idx]].copy()


            noise_row = pd.DataFrame(
                {
                    sensitive_attr: [rng.choice(event_log_model[sensitive_attr].values)],
                    categorical_target: [rng.uniform(0.0, 1.0)],
                },
                index=df_row.index,
            )

            counterfactual_df = gcm.counterfactual_samples(
                causal_model,
                interventions=intervention,
                noise_data=noise_row,
            )

            counterfactual_sensitive_values.append(
                counterfactual_df[sensitive_attr].values[0]
            )
            counterfactual_case_types.append(
                counterfactual_df[categorical_target].values[0]
            )

        df_counterfactual = event_log_case_level.copy()
        df_counterfactual[sensitive_attr] = counterfactual_sensitive_values
        df_counterfactual[categorical_target] = counterfactual_case_types

        counterfactual_map_sensitive = dict(
            zip(event_log_case_level[case_id_col], counterfactual_sensitive_values)
        )
        counterfactual_map_case_type = dict(
            zip(event_log_case_level[case_id_col], counterfactual_case_types)
        )

        event_log_counterfactual[sensitive_attr] = (
            event_log_counterfactual[case_id_col].map(counterfactual_map_sensitive)
        )
        event_log_counterfactual[categorical_target] = (
            event_log_counterfactual[case_id_col].map(counterfactual_map_case_type)
        )
        event_log_counterfactual["counterfactual_index"] = count
        event_logs_counterfactuals.append(event_log_counterfactual)
        count += 1

    event_log_counterfactual = pd.concat(event_logs_counterfactuals, ignore_index=True)

    return event_log_counterfactual

def generate_counterfactuals_dowhy(event_log, sensitive_attrs, labels, case_id_col, dataset_name):
    causal_graph_dependencies = apply_causal_discovery(event_log, labels, case_id_col, setting="fci", alpha=0.01)
    event_log_counterfactual = event_log.copy()
    event_log = event_log.drop_duplicates(subset=[case_id_col], keep="first")
    event_log_copy = event_log.copy()
    event_log_model = event_log[labels].copy()
    dependencies = []
    for dependency in causal_graph_dependencies:
        dependencies.append(dependency)
    causal_graph = nx.DiGraph(dependencies)
    causal_model = gcm.StructuralCausalModel(causal_graph)
    if dataset_name == "bpi12":
        causal_model.set_causal_mechanism(
            "gender",
            gcm.EmpiricalDistribution()
        )
        causal_model.set_causal_mechanism(
            "case:AMOUNT_REQ",
            gcm.AdditiveNoiseModel(
                prediction_model=gcm.ml.create_linear_regressor(),
                noise_model=gcm.ScipyDistribution(norm)
            )
        )
        gcm.fit(causal_model, event_log_model)
        intervention = {
            "gender": lambda g: 2 if g == 1 else 2
        }
        counterfactual_genders = []
        counterfactual_amounts = []
        for idx in range(len(event_log_model)):
            df_row = event_log_model.iloc[[idx]]
            counterfactual_df = gcm.counterfactual_samples(
                causal_model,
                interventions=intervention,
                noise_data=df_row
            )
            counterfactual_genders.append(counterfactual_df["gender"].values[0])
            counterfactual_amounts.append(counterfactual_df["case:AMOUNT_REQ"].values[0])
        df_counterfactual = event_log.copy()
        df_counterfactual["gender"] = counterfactual_genders
        df_counterfactual["case:AMOUNT_REQ"] = counterfactual_amounts
        df_counterfactual["case:AMOUNT_REQ"] = df_counterfactual["case:AMOUNT_REQ"].round().astype(int)
        counterfactual_map_gender = dict(zip(event_log_copy["case_id"], counterfactual_genders))
        counterfactual_map_amount = dict(zip(event_log_copy["case_id"], counterfactual_amounts))
        event_log_counterfactual["gender"] = event_log_counterfactual["case_id"].map(counterfactual_map_gender)
        event_log_counterfactual["case:AMOUNT_REQ"] = event_log_counterfactual["case_id"].map(counterfactual_map_amount)
        event_log_counterfactual["case:AMOUNT_REQ"] = (
            event_log_counterfactual["case:AMOUNT_REQ"] / 1000
        ).round() * 1000
    return event_log_counterfactual

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