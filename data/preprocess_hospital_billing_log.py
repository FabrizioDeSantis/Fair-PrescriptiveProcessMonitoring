from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from counterfactuals import generate_counterfactual_event_log, generate_counterfactuals_dowhy, generate_counterfactuals_dowhy_hospital_billing
from data.utils import remaining_time_helper_func
from transition_system import hash_maps
from data.utils import create_temporal_split, create_prefixes

def print_sensitive_attribute_distribution(data, case_id_column, sensitive_attr):
    data = data[data["gender"]!="non conforming"]
    sensitive_attrs = [sensitive_attr] if isinstance(sensitive_attr, str) else sensitive_attr
    first_occurrences = data.drop_duplicates(subset=[case_id_column], keep="first")

    print("INFO: Sensitive attribute distribution by first case occurrence:")
    for attr in sensitive_attrs:

        if attr not in first_occurrences.columns:
            print(f"WARNING: Sensitive attribute '{attr}' not found. Skipping distribution.")
            continue

        distribution = first_occurrences[attr].value_counts(normalize=False, dropna=False)
        total = distribution.sum()
        print(f"INFO: {attr}")
        for value, percentage in distribution.items():
            value_label = "NaN" if pd.isna(value) else value
            print(f"  {value_label}: {(percentage / total) * 100:.2f}%")

def preprocess_eventlog(data, dataset_name, sensitive_attr, abstraction_mode):

    vocab_sizes = {}
    scalers = {}

    if 'case_id' in data.columns:
        case_id_column = 'case_id'
    elif 'case:concept:name' in data.columns:
        case_id_column = 'case:concept:name'
    else:
        raise ValueError("Could not find case ID column. Available columns: " + str(list(data.columns)))
    
    if 'activity' in data.columns:
        activity_name = 'activity'
    elif 'concept:name' in data.columns:
        activity_name = 'concept:name'
    else:
        raise ValueError("Could not find activity column. Available columns: " + str(list(data.columns)))
    
    data = data.groupby(case_id_column).filter(lambda x: len(x) > 2)

    # Categorize age into <65 and >=65 for computation (kept as binary 0/1).
    # This is applied before any splitting/encoding so it is consistent across all sets.
    if "age" in data.columns:
        age_numeric = pd.to_numeric(data["age"], errors="coerce")
        data = data.copy()
        data["age"] = age_numeric.apply(lambda v: 1 if v >= 65 else (0 if pd.notna(v) else pd.NA)).astype("Int64")

    print_sensitive_attribute_distribution(data, case_id_column, sensitive_attr)
    data = data[data["gender"] != "non conforming"]
    data = remaining_time_helper_func(data, case_id_column)

    train_ids, val_ids, test_ids = create_temporal_split(data, case_id_column, val_ratio=0.2, test_ratio=0.2)
    X_train = data[data[case_id_column].isin(train_ids)]
    X_train.to_csv("hospital_billing_age_train.csv", index=False)  # Save the training data to a CSV file
    X_test = data[data[case_id_column].isin(test_ids)]
    X_test.to_csv("hospital_billing_age_test.csv", index=False)  # Save the test data to a CSV file
    print("INFO: Creating transition system...")
    transition_system = hash_maps.fill_hashmap(X_train=X_train, case_id_name=case_id_column, activity_name=activity_name, thrs=0, abstraction=abstraction_mode, k=None)
    print("INFO: Transition system created.")

    data = data.drop(columns=['lifecycle:transition'])
    categorical_features = [activity_name] + [col for col in data.columns if col not in [case_id_column, activity_name, 'time:timestamp', 'time_delta', 'time_of_day', 'remaining_time', 'time_delta', 'time_of_day']]
    numerical_features = ["time_delta", "time_of_day"]

    for feature in categorical_features:
        data[feature] = pd.Categorical(data[feature])
        if feature == activity_name:
            activity_mapping = dict(enumerate(data[feature].cat.categories, 1))
        data[feature] = data[feature].cat.codes + 1
        vocab_sizes[feature] = data[feature].max()
    
    labels = ['gender', "age", "caseType"]  # Define the labels for counterfactual generations
    print("INFO: Generating counterfactual event log...")
    data_counterfactual = generate_counterfactuals_dowhy_hospital_billing(data, sensitive_attr, labels, case_id_column)
    # convert caseType ("1", "2") to 1 and 2
    data_counterfactual["caseType"] = pd.to_numeric(data_counterfactual["caseType"], errors="coerce")
    print("INFO: Counterfactual event log generated.")

    return create_prefixes(data, data_counterfactual, case_id_column, train_ids, val_ids, test_ids, numerical_features), vocab_sizes, transition_system, activity_mapping