from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from data.utils import remaining_time_helper_func

def create_temporal_split(data, case_id_column, val_ratio=0.2, test_ratio=0.2):
    # data = data.sort_values(by=[case_id_column, 'time:timestamp'])
    # maintain only the first row for each case id
    data = data.groupby(case_id_column).first().reset_index()
    graph_ids = data[case_id_column].unique()
    num_graphs = len(graph_ids)
    test_size = int(num_graphs * test_ratio)
    val_size = int((num_graphs - test_size) * val_ratio)
    train_size = num_graphs - test_size - val_size
    print(f"INFO: Total traces: {num_graphs}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    train_ids = graph_ids[:train_size]
    val_ids = graph_ids[train_size:train_size + val_size]
    test_ids = graph_ids[train_size + val_size:]
    training_data = data[data[case_id_column].isin(train_ids)]
    training_data.to_csv("data/bpi12_dfTrain.csv", index=False)
    val_data = data[data[case_id_column].isin(val_ids)]
    val_data.to_csv("data/bpi12_dfVal.csv", index=False)
    test_data = data[data[case_id_column].isin(test_ids)]
    test_data.to_csv("data/bpi12_dfTest.csv", index=False)
    return train_ids, val_ids, test_ids

def create_ngrams(data, case_id_column, train_ids, val_ids, test_ids, numerical_features, window_size=40):

    ngrams_test = []
    ngrams_val = []
    ngrams_training = []
    labels_training = []
    labels_val = []
    labels_test = []

    scalers_training = {}
    scalers_val = {}
    scalers_test = {}

    max_seq_len = data.groupby(case_id_column).size().max()
    window_size = max_seq_len
    print("INFO: Maximum sequence length: ", max_seq_len)
    training_data = data[data[case_id_column].isin(train_ids)]
    val_data = data[data[case_id_column].isin(val_ids)]
    test_data = data[data[case_id_column].isin(test_ids)]

    for feature in numerical_features:
        scaler = MinMaxScaler()
        training_data[feature] = training_data[feature].ffill()
        training_data[feature] = scaler.fit_transform(training_data[[feature]])
        scalers_training[feature] = scaler
        # scaler = MinMaxScaler()
        val_data[feature] = val_data[feature].ffill()
        val_data[feature] = scaler.transform(val_data[[feature]])
        scalers_val[feature] = scaler
        # scaler = MinMaxScaler()
        test_data[feature] = test_data[feature].ffill()
        test_data[feature] = scaler.transform(test_data[[feature]])
        scalers_test[feature] = scaler

    scaler = StandardScaler()
    training_data["remaining_time"] = scaler.fit_transform(training_data[["remaining_time"]])
    scalers_training["remaining_time"] = scaler
    # scaler = StandardScaler()
    val_data["remaining_time"] = scaler.transform(val_data[["remaining_time"]])
    scalers_val["remaining_time"] = scaler
    # scaler = StandardScaler()
    test_data["remaining_time"] = scaler.transform(test_data[["remaining_time"]])
    scalers_test["remaining_time"] = scaler
    
    for id_value, group in training_data.groupby(case_id_column):

        group = group.reset_index(drop=True)

        group = group.drop(columns=[case_id_column, "time:timestamp"])
        for n in range(1, len(group), 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            ngram_df = ngram_df.drop(columns=["remaining_time"])
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cols]
            ngrams_training.append(cols)

    for id_value, group in val_data.groupby(case_id_column):

        group = group.reset_index(drop=True)

        group = group.drop(columns=[case_id_column, "time:timestamp"])        
        for n in range(1, len(group), 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_val.append(label)
            ngram_df = group.iloc[:n]
            ngram_df = ngram_df.drop(columns=["remaining_time"])
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cols]
            ngrams_val.append(cols)

    for id_value, group in test_data.groupby(case_id_column):

        group = group.reset_index(drop=True)

        group = group.drop(columns=[case_id_column, "time:timestamp"])        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            label = group.iloc[n-1]["remaining_time"]
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            ngram_df = ngram_df.drop(columns=["remaining_time"])
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (max_seq_len-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)
    feature_names = ngram_df.columns.tolist()
    print("INFO: Training samples: ", len(ngrams_training))
    print("INFO: Validation samples: ", len(ngrams_val))
    print("INFO: Test samples: ", len(ngrams_test))

    return ngrams_training, labels_training, ngrams_val, labels_val, ngrams_test, labels_test, max_seq_len, feature_names, scalers_training, scalers_val, scalers_test

def preprocess_eventlog(data, dataset_name, case_id_column):

    vocab_sizes = {}
    scalers = {}

    data = remaining_time_helper_func(data, case_id_column)

    train_ids, val_ids, test_ids = create_temporal_split(data, case_id_column, val_ratio=0.2, test_ratio=0.2)
    
    if dataset_name == "bpi12_kappel":
        data = data.drop(columns=["case:REG_DATE", "lifecycle:transition"])
        categorical_features = ["activity", "org:resource", "gender", "day_of_week"]
        numerical_features = ["case:AMOUNT_REQ", "time_delta", "time_of_day"]

    elif dataset_name == "bpi12":
        data = data.drop(columns=["case:REG_DATE", "lifecycle:transition"])
        categorical_features = ["concept:name", "org:resource", "day_of_week"]
        numerical_features = ["case:AMOUNT_REQ", "time_delta", "time_of_day"]

    for feature in categorical_features:
        data[feature] = pd.Categorical(data[feature])
        data[feature] = data[feature].cat.codes + 1
        vocab_sizes[feature] = data[feature].max()

    return create_ngrams(data, case_id_column, train_ids, val_ids, test_ids, numerical_features), vocab_sizes