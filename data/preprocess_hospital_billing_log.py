from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from counterfactuals import generate_counterfactual_event_log, generate_counterfactuals_dowhy
from data.utils import remaining_time_helper_func
from transition_system import hash_maps
from data.utils import create_temporal_split, create_prefixes

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
    categorical_features = [activity_name] + [col for col in data.columns if col not in [case_id_column, activity_name, 'time:timestamp', 'time_delta', 'time_of_day', 'remaining_time', 'age', 'time_delta', 'time_of_day']]
    numerical_features = ["age", "time_delta", "time_of_day"]

    for feature in categorical_features:
        data[feature] = pd.Categorical(data[feature])
        if feature == activity_name:
            activity_mapping = dict(enumerate(data[feature].cat.categories, 1))
        data[feature] = data[feature].cat.codes + 1
        vocab_sizes[feature] = data[feature].max()

    labels = ['gender', "age"]
    # data["case:AMOUNT_REQ"] = data["case:AMOUNT_REQ"].astype(int)
    print("INFO: Generating counterfactual event log...")
    # if dataset_name == "hospital_billing_age_gender":
    #     data_counterfactual = generate_counterfactuals_dowhy(data, sensitive_attr, labels, case_id_column)
    # else:
    data_counterfactual = generate_counterfactual_event_log(data, sensitive_attr, labels, case_id_column)
    print("INFO: Counterfactual event log generated.")

    return create_prefixes(data, data_counterfactual, case_id_column, train_ids, val_ids, test_ids, numerical_features), vocab_sizes, transition_system, activity_mapping